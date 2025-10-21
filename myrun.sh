# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=2

# python3 bench/bench_qk_int8_pv_fp16_cuda.py

# rm -f qk_int_sv_f16_buffer_cuda.o fused.o debug
nvcc --verbose -arch=sm_86 -G -g -O0 -lineinfo -c -o kv_guided_qk_int8_sv_f16_buffer_cuda.o /home/lin/codes/yf/codes/SageAttention/csrc/qattn/kv_guided_qk_int8_sv_f16_buffer_cuda.cu \
    -I/home/lin/codes/yf/codes/SageAttention/csrc \
    -I/usr/local/cuda-12.1/include \
    -I/home/lin/codes/yf/envs/libtorch/include \
    -I/mnt/sata/anaconda3/include/python3.7m -I/mnt/sata/anaconda3/include/python3.7m \
    -I/home/lin/codes/yf/envs/libtorch/include/torch/csrc/api/include \
    -Xcompiler -fPIC 

# nvcc -arch=sm_86 -c -o fused.o /home/lin/codes/yf/codes/SageAttention/csrc/fused/fused.cu \
#     -I/home/lin/codes/yf/codes/SageAttention/csrc \
#     -I/usr/local/cuda-12.1/include \
#     -I/home/lin/codes/yf/envs/libtorch/include \
#     -I/mnt/sata/anaconda3/include/python3.7m -I/mnt/sata/anaconda3/include/python3.7m \
#     -I/home/lin/codes/yf/envs/libtorch/include/torch/csrc/api/include \
#     -Xcompiler -fPIC 

g++ -v -g -o debug bench/debug.cpp kv_guided_qk_int8_sv_f16_buffer_cuda.o  fused.o  \
    -I/home/lin/codes/yf/codes/SageAttention/csrc \
    -I/usr/local/cuda-12.1/include \
    -I/home/lin/codes/yf/envs/libtorch/include \
    -I/home/lin/codes/yf/envs/libtorch/include/torch/csrc/api/include \
    -L/home/lin/codes/yf/envs/libtorch/lib \
    -L/usr/local/cuda-12.1/lib64 \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lcudart -lpthread -ldl \
    -Wl,-rpath,/home/lin/codes/yf/envs/libtorch/lib -std=c++17

# ./debug