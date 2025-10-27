#!/bin/bash

CUDA_DEVICE=${1:-3}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
# method='xformers, torch, fa2'

echo -e "\033[32m gpu $CUDA_VISIBLE_DEVICES is available in $(date +"%Y-%m-%d %H:%M:%S") \033[0m"
python3 bench/baseline/bench_qk_int8_pv_fp16_triton_baseline.py \
    --method 'fa2' \
    --batch_size 4 \
    --num_heads 32 \
    --head_dim 64 

echo -e "\033[33m quant-pure finish at $(date +"%Y-%m-%d %H:%M:%S") \033[0m"