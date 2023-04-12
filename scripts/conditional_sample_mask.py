"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
直接读取图像，并保存图像的名称(完成)
采样的图像命名对应的名称（完成）
"""

import argparse
import os

import blobfile as bf
import numpy as np
import cv2
import imageio.v2 as imageio
import torch as th
import torch.distributed as dist
from datetime import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.test_script_util import (
    con_model_and_diffusion_defaults,
    con_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir="/home/ubuntu/home/Dammer/guided-diffusion-main/T_V/axia-attention/sample_103")
    date_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    logger.log("start at: ", date_time)

    logger.log("creating model...")
    model, diffusion = con_create_model_and_diffusion(
        **args_to_dict(args, con_model_and_diffusion_defaults().keys())
    )
    # 这里的model加CFG（）
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = con_load_data_for_worker(args.base_samples, args.batch_size)
    name = get_name_list(args.base_samples)

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        # 对条件图像加噪， s取0, 0.1, 0.15, 0.2, 0.3
        if args.gaussion_noise:
            logger.log("对条件图像进行了加噪")
            c = model_kwargs["low_res"]
            c_noise = th.randn_like(c)
            s = [int(0.1 * args.diffusion_steps)] * args.batch_size
            model_kwargs["low_res"] = diffusion.q_sample(c, s, noise=c_noise)

        # 使用无分类引导
        if args.free_CG:
            logger.log("使用了无分类引导")
            sample = diffusion.cfg_sample_loop(
                model,
                (args.batch_size, 3, args.visible_size, args.visible_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        else:
            logger.log("正常的采样策略")
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, args.visible_size, args.visible_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # arr存储所有采样完成的图片
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        for i in range(0, args.num_samples):
            logger.log("save images...")
            imageio.imwrite(logger.get_dir() + '/' + name[i], arr[i])

    dist.barrier()
    date_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    logger.log("sampling complete, end at: ", date_time)


# 这里的base_sample接受一个npz压缩文件，然后第一个文件是包含图像的npy文件
# 直接粗暴的把装有图像的文件转换成一个这样符合输入的npz文件算了。
# 官方的npy大小都是100，128，128，3
def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr01"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


# 改写读取文件的函数
def con_load_data_for_worker(sample_dir, batch_size):
    sample_img = os.listdir(sample_dir)
    sample_img.sort()
    a = len(sample_img)
    # 这里根据图片的大小修改shape
    base_sample = np.ones((a, 256, 256, 3))
    for count in range(0, len(sample_img)):
        name = sample_img[count]
        im_path = os.path.join(sample_dir, name)
        img = imageio.imread(im_path)
        base_sample[count] = img

    image_arr = base_sample
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if len(buffer) == batch_size:
                # 把n个照片转为一个NHWC的tensor：batch
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                yield res
                buffer = []


def get_name_list(sample_dir):
    sample_img = os.listdir(sample_dir)
    sample_img.sort()
    return sample_img


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        gaussion_noise=False,
        free_CG = False,
    )
    defaults.update(con_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
