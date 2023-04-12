export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 16 --num_samples 2160 --timestep_respacing 1000"
nohup python conditional_sample.py $MODEL_FLAGS --model_path ../T_V/mask_train_1/ema_0.9999_320000.pt $SAMPLE_FLAGS --base_samples /home/ubuntu/home/Dammer/TFW/test/gray_crop_256x256

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 32 --num_samples 2160 --timestep_respacing 1000"
python scripts/conditional_sample_mask.py $MODEL_FLAGS --model_path ./T_V/gaussion_train/ema_0.9999_320000.pt $SAMPLE_FLAGS --base_samples /home/ubuntu/home/Dammer/TFW/test/gray_crop_256x256

# 这里尝试FCG，只采样100张
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --visible_size 256 --thermal_size 256"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 1000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 32 --num_samples 32 --timestep_respacing 1000"
python scripts/conditional_sample_mask.py $MODEL_FLAGS --model_path ./T_V/axia-attention/train/ema_0.9999_103000.pt $SAMPLE_FLAGS --base_samples /home/ubuntu/home/Dammer/TFW/test/gray_crop_256x256
