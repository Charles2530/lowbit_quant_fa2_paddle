#!/bin/bash

CUDA_DEVICE=${1:-3}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
# method='xformers, torch,fa2'

echo -e "\033[32m gpu $CUDA_VISIBLE_DEVICES is available in $(date +"%Y-%m-%d %H:%M:%S") \033[0m"
# TRITON_INTERPRETER=1 python3 bench/quant/bench_int4_triton.py \
python3 bench/quant/bench_int4_triton.py \
    --method 'torch' \
    --batch_size 4 \
    --num_heads 12 \
    --head_dim 128 

echo -e "\033[33m quant-pure finish at $(date +"%Y-%m-%d %H:%M:%S") \033[0m"