MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1  --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 16 --lr 1e-4 --save_interval 2000 --weight_decay 0.05"
SAMPLE_FLAGS="--batch_size 4 --timestep_respacing 1000"
python conditional_sample.py $MODEL_FLAGS --model_path ../T_V/ema_0.9999_460000.pt $SAMPLE_FLAGS --base_sampel ../test1.npy