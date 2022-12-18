#!/bin/sh
export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 2 --learn_sigma True --dropout 0.3"
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 64 --use_fp16 true --max_steps 5000 --save_interval 2500"
