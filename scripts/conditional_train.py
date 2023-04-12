"""
Train a condition model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.con_image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
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
    logger.configure(dir='/home/ubuntu/home/Dammer/guided-diffusion-main/T_V/TFW_train_1')

    logger.log("creating model...")
    # 这里创造的模型接口不同，使用的超分接口
    # 扩散还是那个扩散，只是模型略有不同
    # 这里全部沿用超分模型的接口
    model, diffusion = con_create_model_and_diffusion(
        **args_to_dict(args, con_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data中第一个是可见光图像，第二个是一个字典，其中low_res指向热图
    data = load_condition_data(
        args.visible_dir,
        args.thermal_dir,
        args.batch_size,
        image_size=256,
    )

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
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_condition_data(visible_dir, thermal_dir, batch_size, image_size):
    data = load_data(
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
        thermal_dir="",
        visible_dir="",
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
    )
    defaults.update(con_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
