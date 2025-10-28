#!/bin/bash
# Triton 环境设置脚本
# 用于解决 Triton 驱动问题和 PaddlePaddle 兼容性问题

CUDA_DEVICE=${1:-0}

echo "Setting up Triton environment for GPU $CUDA_DEVICE..."

# 设置环境路径（不依赖conda activate，直接使用conda环境路径）
CONDA_ENV_PATH=/mnt/disk3/conda/envs/paddle_nightly_bk
conda activate $CONDA_ENV_PATH
# 设置 Python 环境
export PATH=$CONDA_ENV_PATH/bin:$PATH

# 设置 CUDA 环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH

# 设置 cuDNN 库路径
export LD_LIBRARY_PATH=$CONDA_ENV_PATH/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 设置 Triton 相关环境变量
export TRITON_CACHE_DIR=/tmp/triton_cache_${CUDA_DEVICE}
export TRITON_DEBUG=0
export TRITON_DISABLE_LINE_INFO=1

# 创建 Triton 缓存目录
mkdir -p $TRITON_CACHE_DIR

# 设置 Python 路径 - 使用当前工作目录
export PYTHONPATH=$(pwd):$PYTHONPATH

# 设置 PaddlePaddle 相关环境变量
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

# 验证环境
echo "Verifying environment..."
echo "  - Python: $(which python3)"
echo "  - Pip: $(which pip)"
echo "  - Triton version: $(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'Not available')"
echo "  - Paddle version: $(python3 -c 'import paddle; print(paddle.__version__)' 2>/dev/null || echo 'Not available')"

echo "✓ Environment setup complete!"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - CUDA_HOME: $CUDA_HOME"
echo "  - CUDA_PATH: $CUDA_PATH"
echo "  - TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "  - PYTHONPATH: $PYTHONPATH"
