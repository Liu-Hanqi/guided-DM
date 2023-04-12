export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 4 --timestep_respacing 1000"
python conditional_train.py --thermal_dir /home/ubuntu/home/Dammer/T_V_Dataset/TUfts/train-RGB-faces-128x128 --visible_dir /home/ubuntu/home/Dammer/T_V_Dataset/TUfts/train-thermal-face-128x128  $MODEL_FLAGS $SAMPLE_FLAGS