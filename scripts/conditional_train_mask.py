"""
Train a condition model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.con_image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.test_script_util import (
    con_model_and_diffusion_defaults,
    con_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    logger.configure(dir='/home/ubuntu/home/Dammer/guided-diffusion-main/T_V/axia-attention/train')

    logger.log("creating model...")
    # 这里创造的模型接口不同，使用的超分接口
    # 扩散还是那个扩散，只是模型略有不同
    # 这里全部沿用超分模型的接口
    model, diffusion = con_create_model_and_diffusion(
        **args_to_dict(args, con_model_and_diffusion_defaults().keys())
    )
    print(model)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data中第一个是可见光图像，第二个是一个字典，其中low_res指向热图
    print("mask_size:", args.mask_size)
    print("gaussian_noise:", args.gaussian_noise)
    print("selective_conditon:", args.selective_conditon)
    print("use_vgg:", args.use_vgg)
    print("use_gan:", args.use_gan)

    data = load_condition_data(
        args.mask_size,
        args.visible_dir,
        args.thermal_dir,
        args.batch_size,
        image_size=256,
    )

    if len(args.val_thermal_dir) and len(args.val_visible_dir) != 0:
        val_data = load_condition_data(
            0,
            args.val_visible_dir,
            args.val_thermal_dir,
            args.batch_size//2,
            image_size=256,
        )
    else: val_data=None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        val_data=val_data,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        gaussian_noise = args.gaussian_noise,
        selective_conditon = args.selective_conditon,
        use_vgg=args.use_vgg,
        use_gan=args.use_gan,
    ).run_loop()


def load_condition_data(mask_size, visible_dir, thermal_dir, batch_size, image_size):
    data = load_data(
        mask_size = mask_size,
        the_dir=thermal_dir,
        vis_dir=visible_dir,
        batch_size=batch_size,
        image_size=image_size,
    )
    # 把低分辨率的图片放在model_kwargs的字典中，key值为：low_res。
    # 低分辨率的图片是高分辨率图片压缩的，因此当时是成对的图片
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        thermal_dir="/home/ubuntu/home/Dammer/TFW/train/gray_crop_256x256",
        visible_dir="/home/ubuntu/home/Dammer/TFW/train/rgb_crop_256x256",
        val_thermal_dir="", # 加入val数据监控
        val_visible_dir="", # 加入val数据监控
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        mask_size = 32, # 改进1：加随机mask
        gaussian_noise = False, # 改进2：加高斯噪声
        selective_conditon = False, # 在训练时，有一些步骤加入条件，一些步骤不加入条件，比如在一次训练中有10%的可能性不加入条件图像。
        use_vgg=True, # 在训练时候，加入感知损失
        use_gan=True, # 在训练时候，加入对抗损失
    )

    defaults.update(con_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
