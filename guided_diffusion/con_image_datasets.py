import math
import random
import torch as th
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.distributed as dist
import os


def load_data(
        *,
        mask_size, # 尝试加mask区域
        the_dir,
        vis_dir,
        batch_size,
        image_size,
        deterministic=False,
        random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param the_dir: a thermal dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    print(the_dir)
    print(vis_dir)
    print(mask_size)
    if not the_dir:
        raise ValueError("unspecified data directory")
    # _list_image_files_recursively这个函数本身就自带排序哈
    the_files, the_name = _list_image_files_recursively(the_dir)
    vis_files, vis_name = _list_image_files_recursively(vis_dir)

    dataset = ImageDataset(
        mask_size,
        image_size,
        the_files,
        vis_files,
        the_name,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(the_dir):
    results = []
    name = []
    for entry in sorted(bf.listdir(the_dir)):
        full_path = bf.join(the_dir, entry)
        name.append(entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results, name


class RandomCrop(object):

    def __init__(self, crop_size=[256, 128]):
        """Set the height and weight before and after cropping"""
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, inputs, target):
        input_size_h, input_size_w, _ = inputs.shape
        try:
            x_start = random.randint(0, input_size_w - self.crop_size_w)
            y_start = random.randint(0, input_size_h - self.crop_size_h)
            inputs = inputs[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
            target = target[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        except:
            inputs = cv2.resize(inputs, (128, 128))
            target = cv2.resize(target, (128, 128))

        return inputs, target


class ImageDataset(Dataset):
    def __init__(
            self,
            mask_size,  # 尝试加mask
            resolution,
            the_paths,
            vis_paths,
            name,
            shard=0,
            num_shards=1,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        # 数组a[0:][::1]的意思是从0位置开始，每1步取出一个数
        self.the_images = the_paths[shard:][::num_shards]
        self.vis_images = vis_paths[shard:][::num_shards]
        self.random_flip = random_flip
        self.name = name[shard:][::num_shards]
        self.mask_size = mask_size

    def __len__(self):
        return len(self.the_images)

    def __getitem__(self, idx):
        # the_path和vis_path就是存储的图像路径
        the_path = self.the_images[idx]
        vis_path = self.vis_images[idx]
        name = self.name[idx]

        # 在这里加mask，这个mask时固定的。
        # 热图带有mask（ok）
        # 热图和可见光同时带有mask的读取条件（ok）
        if self.mask_size != 0:
            # print("对图像添加了固定的mask")
            mask = np.ones([256, 256]).astype(np.float32)
            mask_x = random.randint(0, (256 - 32))
            mask_y = random.randint(0, (256 - 32))
            mask[mask_y:mask_y + self.mask_size, mask_x:mask_x + self.mask_size] = 0.
            with bf.BlobFile(the_path, "rb") as f:
                thermal_image = self.process_and_load_mask_images(f, mask)
            with bf.BlobFile(vis_path, "rb") as f1:
                visible_image = self.process_and_load_images(f1)

        else:
            # 无任何条件的读取图片
            # print("没有添加mask")
            with bf.BlobFile(the_path, "rb") as f:
                thermal_image = self.process_and_load_images(f)
            with bf.BlobFile(vis_path, "rb") as f1:
                visible_image = self.process_and_load_images(f1)

        out_dict = {}
        out_dict["low_res"] = thermal_image
        
        return visible_image, out_dict

    def process_and_load_images(self, path):
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.resize((self.resolution, self.resolution))
        arr = np.array(pil_image).astype(np.float32)
        arr = arr / 127.5 - 1.0 # 数组的值为[-1, 1]
        arr = np.transpose(arr, [2, 0, 1])
        # arr.shape = (3, 256, 256)

        return arr

    def process_and_load_mask_images(self, path, mask):
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.resize((self.resolution, self.resolution))
        arr = np.array(pil_image).astype(np.float32)
        arr = arr / 127.5 - 1.0  # 数组的值为[-1, 1]
        arr = np.transpose(arr, [2, 0, 1])
        r = arr[0, :, :] * mask
        b = arr[1, :, :] * mask
        g = arr[2, :, :] * mask
        arr = cv2.merge((r, b, g))
        arr = np.transpose(arr, [2, 0, 1])
        # arr.shape = (3, 256, 256)

        return arr
