CUDA_DEVICE=${1:-1}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
# method='xformers, torch,fa2'

echo -e "\033[32m gpu $CUDA_VISIBLE_DEVICES is available in $(date +"%Y-%m-%d %H:%M:%S") \033[0m"
python example/test_sageattn_operator.py\
    --op int8 \
    --batch_size 4 \
    --num_heads 32 \
    --seq_len 1024 \
    --head_dim 64 \
    --repeats 5

python example/test_sageattn_operator.py\
    --op int4 \
    --batch_size 4 \
    --num_heads 32 \
    --seq_len 1024 \
    --head_dim 64 \
    --repeats 5

echo -e "\033[33m quant-pure finish at $(date +"%Y-%m-%d %H:%M:%S") \033[0m"

