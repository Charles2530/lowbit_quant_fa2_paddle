CUDA_DEVICE=${1:-3}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo -e "\033[32m gpu $CUDA_VISIBLE_DEVICES is available in $(date +"%Y-%m-%d %H:%M:%S") \033[0m"
python bench/video_test/sageattn_cogvideo_int8.py --compile

echo -e "\033[33m quant-pure finish at $(date +"%Y-%m-%d %H:%M:%S") \033[0m"