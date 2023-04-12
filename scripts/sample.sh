MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 64 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --learn_sigma True --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--batch_size 32 --lr 3e-5 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 1000"
python image_sample.py --model_path ../128classifier-log/openai-2022-10-29-10-59-18-285633/ema_0.9999_210000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SIMPLE_FLAGS
