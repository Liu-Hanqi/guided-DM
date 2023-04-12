export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 16 --timestep_respacing 1000"
python conditional_train.py --thermal_dir /home/ubuntu/home/Dammer/TFW/train/gray_crop_256x256 --visible_dir /home/ubuntu/home/Dammer/TFW/train/rgb_crop_256x256  $MODEL_FLAGS $SAMPLE_FLAGS

cd home/Dammer/guided-diffusion-main
source ~/.bashrc

# mask的命令
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 8 --lr 2e-5 --save_interval 1000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 8 --timestep_respacing 1000"
nohup python scripts/conditional_train_mask.py --thermal_dir /home/ubuntu/home/Dammer/TFW/train/gray_crop_256x256 --visible_dir /home/ubuntu/home/Dammer/TFW/train/rgb_crop_256x256 $MODEL_FLAGS $TRAIN_FLAGS

# 高斯噪声的命令
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 64 --timestep_respacing 1000"
nohup python scripts/conditional_train_mask.py --thermal_dir /home/ubuntu/home/Dammer/TFW/train/gray_crop_256x256 --visible_dir /home/ubuntu/home/Dammer/TFW/train/rgb_crop_256x256
