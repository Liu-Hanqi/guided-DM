"""
需要的指标有：
    python evaluations/T_V_evaluator.py
    1. PSNR(ok), SSIM(ok), Deg.(), (小)LPIPS(ok)
    2. Rank-1(), VR@FAR=1%(), VR@FAR=0.1%()
"""
import numpy as np
import os
import imageio.v2 as imageio
import lpips
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def main():
    the2vis_dir = '/home/ubuntu/home/Dammer/guided-diffusion-main/T_V/axia-attention/sample_103'
    visible_dir = '/home/ubuntu/home/Dammer/TFW/test/rgb_crop_256x256'

    # t2v_array: 图像转换后的数组(NHWC)
    # vis_array: 真实可见光的数组(NHWC)
    # numbers: 图像转换的数量
    t2v_array, vis_array, numbers = sort_load_image(the2vis_dir, visible_dir)
    calculate_psnr(t2v_array, vis_array, numbers)
    calculate_ssim(t2v_array, vis_array, numbers)
    calculate_lpips(t2v_array, vis_array, numbers)
    # 计算FID直接使用提供的包，官方说如果数据少于2048，可以改模型，但是最好不要
    # python -m pytorch_fid 数据集1 数据集2
    # python -m pytorch_fid /home/ubuntu/home/Dammer/TFW/test/rgb_crop_256x256 /home/ubuntu/home/Dammer/guided-diffusion-main/T_V/mask+vgg+adv_test/370test

def sort_load_image(the2vis_dir, visible_dir):
    t2v_list = os.listdir(the2vis_dir)
    vis_list = os.listdir(visible_dir)
    t2v_list.sort()
    vis_list.sort()
    t2v_img = []
    vis_img = []
    image_len = len(t2v_list)
    print(image_len)
    for i in range(0, image_len):
        subimg1 = t2v_list[i]
        t2v_subimg = os.path.join(the2vis_dir, subimg1)
        t2v_img.append(t2v_subimg)
        subimg2 = vis_list[i]
        if(subimg1 != subimg2):print("name wrong!")
        vis_subimg = os.path.join(visible_dir, subimg2)
        vis_img.append(vis_subimg)

    temp= imageio.imread(t2v_img[0])
    h = temp.shape[0]
    w = temp.shape[1]
    c = temp.shape[2]
    a = np.ones((image_len, h, w, c))
    b = np.ones((image_len, h, w, c))
    for i in range(0, image_len):
        img = imageio.imread(t2v_img[i])
        img2 = imageio.imread(vis_img[i])
        a[i] = img
        b[i] = img2
    return a, b, image_len

def calculate_psnr(arr1, arr2, num):
    psnr_mean = 0
    for count in range(0, num):
        psnr = compare_psnr(arr1[count], arr2[count], data_range=255)
        psnr_mean = psnr_mean + psnr

    print('mean psnr:{}'.format(psnr_mean / num))

def calculate_ssim(arr1, arr2, num):
    psnr_ssim = 0
    for count in range(0, num):
        ssim = compare_ssim(arr1[count], arr2[count], data_range=255, channel_axis=2)
        psnr_ssim = psnr_ssim + ssim

    print('mean ssim:{}'.format(psnr_ssim / num))

def calculate_lpips(arr1, arr2, num):
    lpips_mean = 0
    loss_fn = lpips.LPIPS(net='alex')

    for i in range(0, num):
        temp1 = (arr1[i] / (255. / 2.) - 1.)[ np.newaxis, :, :, :].transpose((0, 3, 1, 2))
        temp1 = torch.from_numpy(temp1).float()
        temp2 = (arr2[i] / (255. / 2.) - 1.)[ np.newaxis, :, :, :].transpose((0, 3, 1, 2))
        temp2 = torch.from_numpy(temp2).float()
        lpip = loss_fn.forward(temp1, temp2)
        lpips_mean = lpips_mean + lpip

    lpips_mean=(lpips_mean / num).item()
    print('mean lpips:{}'.format(lpips_mean))


if __name__ == "__main__":
    main()