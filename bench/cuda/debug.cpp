#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Function declaration for the kernel
// torch::Tensor qk_int8_sv_f16_accum_f16_attn_buf(
//     torch::Tensor query,
//     torch::Tensor key,
//     torch::Tensor value,
//     torch::Tensor output,
//     torch::Tensor query_scale,
//     torch::Tensor key_scale,
//     int tensor_layout,
//     int is_causal,
//     int qk_quant_gran,
//     float sm_scale,
//     int return_lse);

// Function declaration for the kernel
torch::Tensor kv_guided_qk_int8_sv_f16_accum_f16_attn_buf(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);


// Function declaration for the kernel
void quant_per_block_int8_fuse_sub_mean_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int tensor_layout);
    
void quant_per_block_int8_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    float sm_scale,
    int block_size,
    int tensor_layout);

// void benchmark_kernel(int B, int N, int H, int D, int tensor_layout, int block_size, int is_causal, int qk_quant_gran, int return_lse) {
//     // Input dimensions
//     int head_dim = D;
//     const int QN = 1; 
    
//     // Input dimensions
//     float sm_scale = 1.0f; 

//     // Initialize tensors
//     auto query_f16 = torch::randn({B, QN, H, D}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
//     auto key_f16 = torch::randn({B, N, H, D}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
//     auto value = torch::randn({B, N, H, D}, torch::dtype(torch::kHalf).device(torch::kCUDA));

//     auto km = key_f16.mean(1); // Mean along dimension 1 (N)

//     auto query_int8 = torch::empty({B, QN, H, D}, torch::dtype(torch::kInt8).device(torch::kCUDA));
//     auto key_int8 = torch::empty({B, N, H, D}, torch::dtype(torch::kInt8).device(torch::kCUDA));
//     // Compute shape of query_scale and key_scale based on quantization granularity
//     torch::Tensor query_scale;
//     torch::Tensor key_scale;
    
//     int WARP_Q = 32;
//     int WARP_K = 64;

//     if (qk_quant_gran == 2) { // Per warp
//         query_scale = torch::randn({B, H, N / WARP_Q }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         key_scale = torch::randn({B, H, N / WARP_K }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         // query_scale = torch::empty({B, H,  (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         // key_scale = torch::empty({B, H, (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//     } else if (qk_quant_gran == 3) { // Per thread
//         query_scale = torch::randn({B, H, QN / WARP_Q * 8}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         key_scale = torch::randn({B, H, N / WARP_K * 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         // query_scale = torch::empty({B, H,  (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//         // key_scale = torch::empty({B, H, (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//     }

//     auto q_scale = torch::empty({B, H, (QN + block_size - 1) / block_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//     auto k_scale = torch::empty({B, H, (N + block_size - 1) / block_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
//     auto output = torch::empty({B, N, H, D}, torch::dtype(torch::kHalf).device(torch::kCUDA));

//     // Call the kernel
//     quant_per_block_int8_cuda(query_f16, query_int8, q_scale, sm_scale, block_size, tensor_layout);
//     quant_per_block_int8_fuse_sub_mean_cuda(key_f16, km, key_int8, k_scale, block_size, tensor_layout);

//     // std::cout << "query_scale shape: " << query_scale.sizes() << std::endl; 
//     // std::cout << "key_scale shape: " << key_scale.sizes() << std::endl;

//     std::cout << "Data init finish" << std::endl;

//     // Warm-up kernel to avoid JIT overhead
//     int warmup_iter = 100; 
//     int iter = 100;
//     for (int i = 0; i < warmup_iter; ++i) {
//         qk_int8_sv_f16_accum_f16_attn_buf(query_int8, key_int8, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
//     }

//     std::cout << "Warm-up kernel finish"  << std::endl;

//     // Measure kernel execution time
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     for (int i = 0; i < iter; ++i) { // Run the kernel 100 times for averaging
//         qk_int8_sv_f16_accum_f16_attn_buf(query_int8, key_int8, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     std::cout << "Average execution time: " << (milliseconds / 100) << " ms (avg of "<< iter <<" times)" << std::endl;

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     // Optionally, validate output if a baseline is available
//     // auto lse = qk_int8_sv_f16_accum_f16_attn_buf(query, key, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
//     // Validate `lse` here if needed
// }

void benchmark_ours_kernel(int B, int N, int H, int D, int tensor_layout, int block_size, int is_causal, int qk_quant_gran, int return_lse) {
    // Input dimensions
    int head_dim = D;
    
    // Input dimensions
    float sm_scale = 1.0f; 

    // Initialize tensors
    auto query_f16 = torch::randn({B, 1, H, D}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto key_f16 = torch::randn({1, N, H, D}, torch::dtype(torch::kFloat16).device(torch::kCUDA));  // 1, 128, 32, 64
    auto value = torch::randn({1, N, H, D}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    auto km = key_f16.mean(1); // Mean along dimension 1 (N)

    auto query_int8 = torch::empty({B, 1, H, D}, torch::dtype(torch::kInt8).device(torch::kCUDA));
    auto key_int8 = torch::empty({1, N, H, D}, torch::dtype(torch::kInt8).device(torch::kCUDA));
    // Compute shape of query_scale and key_scale based on quantization granularity
    torch::Tensor query_scale;
    torch::Tensor key_scale;
    
    int WARP_Q = 32;
    int WARP_K = 64;

    if (qk_quant_gran == 2) { // Per warp
        // query_scale = torch::randn({B, H, N / WARP_Q }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        query_scale = torch::randn({B, H, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        key_scale = torch::randn({1, H, N / WARP_K }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        // query_scale = torch::empty({B, H,  (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        // key_scale = torch::empty({B, H, (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    } else if (qk_quant_gran == 3) { // Per thread
        query_scale = torch::randn({B, H, N / WARP_Q * 8}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        key_scale = torch::randn({1, H, N / WARP_K * 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        // query_scale = torch::empty({B, H,  (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        // key_scale = torch::empty({B, H, (N + block_size - 1) / block_size }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }

    // auto q_scale = torch::empty({B, H, (N + block_size - 1) / block_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto q_scale = torch::empty({B, H, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto k_scale = torch::empty({1, H, (N + block_size - 1) / block_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = torch::empty({B, 1, H, D}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    // Call the kernel
    quant_per_block_int8_cuda(query_f16, query_int8, q_scale, sm_scale, block_size, tensor_layout);
    quant_per_block_int8_fuse_sub_mean_cuda(key_f16, km, key_int8, k_scale, block_size, tensor_layout);

    // std::cout << "query_scale shape: " << query_scale.sizes() << std::endl; 
    // std::cout << "key_scale shape: " << key_scale.sizes() << std::endl;

    std::cout << "Data init finish" << std::endl;

    // Warm-up kernel to avoid JIT overhead
    int warmup_iter = 1; 
    int iter = 1;
    for (int i = 0; i < warmup_iter; ++i) {
        kv_guided_qk_int8_sv_f16_accum_f16_attn_buf(query_int8, key_int8, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
    }

    std::cout << "Warm-up kernel finish"  << std::endl;

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iter; ++i) { // Run the kernel 100 times for averaging
        kv_guided_qk_int8_sv_f16_accum_f16_attn_buf(query_int8, key_int8, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Average execution time: " << (milliseconds / 100) << " ms (avg of "<< iter <<" times)" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}



int main() {
    std::srand(42); // 固定种子为42
    torch::manual_seed(42); // 固定种子为42

    int B = 8;               // Batch size
    int N = 128;              // Sequence length
    int H = 32;                // Number of heads
    int D = 64;               // Head dimension
    int tensor_layout = 0;    // Layout: 0 for [B, N, H, D], 1 for [B, H, N, D]
    int is_causal = 0;        // Non-causal mode
    int qk_quant_gran = 2;    // Quant granularity, 2: per warp / 3 per thread
    int return_lse = 0;       // No LSE return
    int block_size = 128;       // 64/128

    const char* c_tensor_layout = nullptr;
    const char* c_qk_quant_gran = nullptr;
    if (tensor_layout == 0) {
        c_tensor_layout = "[B, N, H, D]";
    } else {
        c_tensor_layout = "[B, H, N, D]";
    } 
    if (qk_quant_gran == 2) {
        c_qk_quant_gran = "per warp";
    } else {
        c_qk_quant_gran = "per thread";
    }
    std::cout << "\n\nBegin testing...\n\n" << std::endl;
    std::cout <<  "B: " << B << ", N: " << N << ", H: " << H << ", D: " << D << ", tensor_layout: " << c_tensor_layout << ", is_causal: " << is_causal << ", qk_quant_gran: " << c_qk_quant_gran << ", return_lse: " << return_lse << ", block_size: " << block_size << std::endl;
    
    // benchmark_kernel(B, N, H, D, tensor_layout, block_size, is_causal, qk_quant_gran, return_lse);
    benchmark_ours_kernel(B, N, H, D, tensor_layout, block_size, is_causal, qk_quant_gran, return_lse);
    return 0;
}