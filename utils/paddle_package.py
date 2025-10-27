import time
import paddle
import numpy as np
from typing import Callable, Any, Tuple, Optional


def benchmark_forward(
    forward_func: Callable,
    *args,
    repeats: int = 100,
    verbose: bool = False,
    desc: str = "Benchmark",
    **kwargs
) -> Tuple[Any, np.ndarray]:
    """
    Custom benchmark function to replace flash_attn.utils.benchmark.benchmark_forward.
    
    Args:
        forward_func (Callable): The function to benchmark.
        *args: Positional arguments to pass to forward_func.
        repeats (int): Number of repetitions for benchmarking. Defaults to 100.
        verbose (bool): Whether to print verbose output. Defaults to False.
        desc (str): Description for the benchmark. Defaults to "Benchmark".
        **kwargs: Keyword arguments to pass to forward_func.
        
    Returns:
        Tuple[Any, np.ndarray]: Tuple containing the output of forward_func and timing results.
    """
    # Warmup runs
    for _ in range(5):
        try:
            _ = forward_func(*args, **kwargs)
        except Exception as e:
            if verbose:
                print(f"Warning: Warmup failed with error: {e}")
    
    # Synchronize before timing
    paddle.device.synchronize()
    
    # Timing runs
    times = []
    for i in range(repeats):
        start_time = time.perf_counter()
        try:
            output = forward_func(*args, **kwargs)
            paddle.device.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            if verbose:
                print(f"Warning: Run {i} failed with error: {e}")
            times.append(float('inf'))
    
    times = np.array(times)
    
    if verbose:
        print(f"{desc}: Mean time: {times.mean()*1000:.2f}ms, "
              f"Std: {times.std()*1000:.2f}ms, "
              f"Min: {times.min()*1000:.2f}ms, "
              f"Max: {times.max()*1000:.2f}ms")
    
    return output, times


def benchmark_attention_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Custom benchmark function for scaled dot-product attention.
    
    Args:
        forward_func (Callable): Function implementing scaled dot-product attention.
        seq_lens (list): List of sequence lengths to benchmark.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        repeats (int): Number of repetitions for benchmarking. Defaults to 100.
        causal (bool): Whether attention is causal. Defaults to False.
        logger (Optional[Any]): Logger to log benchmarking results. Defaults to None.
    """
    for seq_len in seq_lens:
        flops = (
            4
            * num_heads
            * batch_size
            * head_dim
            * seq_len
            * seq_len
            // (2 if causal else 1)
        )
        
        q = paddle.randn(batch_size, num_heads, seq_len, head_dim).half().cuda()
        k = paddle.randn(batch_size, num_heads, seq_len, head_dim).half().cuda()
        v = paddle.randn(batch_size, num_heads, seq_len, head_dim).half().cuda()
        
        _, time_results = benchmark_forward(
            forward_func,
            q, k, v,
            is_causal=causal,
            repeats=repeats,
            verbose=False,
            desc="Custom Attention"
        )
        
        if logger:
            logger.log(
                f"{seq_len} flops: {flops / time_results.mean * 1e-12:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s"
            )
        else:
            print(
                f"{seq_len} flops: {flops / time_results.mean * 1e-12:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s"
            )


def benchmark_triton_attention_int8_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    q_dtype: paddle.dtype = paddle.float16,
    k_dtype: paddle.dtype = paddle.float16,
    v_dtype: paddle.dtype = paddle.float16,
    output_dtype: paddle.dtype = paddle.bfloat16,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Custom benchmark function for Triton attention with int8 quantization.
    
    Args:
        forward_func (Callable): The forward function to benchmark.
        seq_lens (list): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        q_dtype (paddle.dtype): Data type for query tensor.
        k_dtype (paddle.dtype): Data type for key tensor.
        v_dtype (paddle.dtype): Data type for value tensor.
        output_dtype (paddle.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        logger (Optional[Any]): Logger to log benchmarking results.
    """
    from src.triton.quant_per_block import per_block_int8
    
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
            
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=q_dtype,
        )
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=k_dtype,
        )
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        
        q_codes, q_scale, k_codes, k_scale = per_block_int8(prepare_q, prepare_k)
        
        _, time_results = benchmark_forward(
            forward_func,
            q_codes, k_codes, v, q_scale, k_scale,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Int8"
        )
        
        flops_per_time = flops / time_results.mean * 1e-12
        q_output, _ = forward_func(
            q_codes, k_codes, v, q_scale, k_scale, output_dtype=output_dtype
        )
        
        target = paddle.nn.functional.scaled_dot_product_attention(
            prepare_q.transpose([0, 2, 1, 3]),
            prepare_k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        
        loss_fn = paddle.nn.MSELoss()
        loss = loss_fn(q_output, target)
        
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )


def benchmark_triton_attention_int4_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    q_dtype: paddle.dtype = paddle.float16,
    k_dtype: paddle.dtype = paddle.float16,
    v_dtype: paddle.dtype = paddle.float16,
    output_dtype: paddle.dtype = paddle.float16,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Custom benchmark function for Triton attention with int4 quantization.
    
    Args:
        forward_func (Callable): The forward function to benchmark.
        seq_lens (list): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        q_dtype (paddle.dtype): Data type for query tensor.
        k_dtype (paddle.dtype): Data type for key tensor.
        v_dtype (paddle.dtype): Data type for value tensor.
        output_dtype (paddle.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        logger (Optional[Any]): Logger to log benchmarking results.
    """
    from src.triton.quant_per_block import per_block_int4
    
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
            
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=q_dtype,
        )
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=k_dtype,
        )
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        
        q_codes, q_scale, k_codes, k_scale = per_block_int4(prepare_q, prepare_k)
        
        _, time_results = benchmark_forward(
            forward_func,
            q_codes, q_scale, k_codes, k_scale, v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Int4"
        )
        
        flops_per_time = flops / time_results.mean * 1e-12
        output, lse = forward_func(
            q_codes, q_scale, k_codes, k_scale, v, output_dtype=output_dtype
        )
        
        target = paddle.nn.functional.scaled_dot_product_attention(
            prepare_q.transpose([0, 2, 1, 3]),
            prepare_k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        
        loss_fn = paddle.nn.MSELoss()
        try:
            loss = loss_fn(output, target)
        except:
            loss = paddle.inf
        
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )


def benchmark_triton_attention_q_int8_k_int4_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    q_dtype: paddle.dtype = paddle.float16,
    k_dtype: paddle.dtype = paddle.float16,
    v_dtype: paddle.dtype = paddle.float16,
    output_dtype: paddle.dtype = paddle.float16,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Custom benchmark function for Triton attention with mixed q_int8_k_int4 quantization.
    
    Args:
        forward_func (Callable): The forward function to benchmark.
        seq_lens (list): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        q_dtype (paddle.dtype): Data type for query tensor.
        k_dtype (paddle.dtype): Data type for key tensor.
        v_dtype (paddle.dtype): Data type for value tensor.
        output_dtype (paddle.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        logger (Optional[Any]): Logger to log benchmarking results.
    """
    from src.triton.quant_per_block import per_block_q_int8_k_int4
    
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
            
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=q_dtype,
        )
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=k_dtype,
        )
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        
        q_codes, q_scale, k_codes, k_scale = per_block_q_int8_k_int4(prepare_q, prepare_k)
        
        _, time_results = benchmark_forward(
            forward_func,
            q_codes, q_scale, k_codes, k_scale, v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Q_Int8_K_Int4"
        )
        
        flops_per_time = flops / time_results.mean * 1e-12
        output, lse = forward_func(
            q_codes, q_scale, k_codes, k_scale, v, output_dtype=output_dtype
        )
        
        target = paddle.nn.functional.scaled_dot_product_attention(
            prepare_q.transpose([0, 2, 1, 3]),
            prepare_k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        
        loss_fn = paddle.nn.MSELoss()
        try:
            loss = loss_fn(output, target)
        except:
            loss = paddle.inf
        
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )


def benchmark_triton_attention_int2_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    q_dtype: paddle.dtype = paddle.float16,
    k_dtype: paddle.dtype = paddle.float16,
    v_dtype: paddle.dtype = paddle.float16,
    output_dtype: paddle.dtype = paddle.float16,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Custom benchmark function for Triton attention with int2 quantization.
    
    Args:
        forward_func (Callable): The forward function to benchmark.
        seq_lens (list): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        q_dtype (paddle.dtype): Data type for query tensor.
        k_dtype (paddle.dtype): Data type for key tensor.
        v_dtype (paddle.dtype): Data type for value tensor.
        output_dtype (paddle.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        logger (Optional[Any]): Logger to log benchmarking results.
    """
    from src.triton.utils.quant.new_pack import triton_quantize_and_pack_along_last_dim
    
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
            
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=q_dtype,
        )
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=k_dtype,
        )
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        
        q_codes, q_scale, q_mn = triton_quantize_and_pack_along_last_dim(
            prepare_q, group_size=32, bit=8
        )
        k_codes, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
            prepare_k, group_size=32, bit=2
        )
        
        _, time_results = benchmark_forward(
            forward_func,
            q_codes, q_scale, q_mn, k_codes, k_scale, k_mn, v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Int2"
        )
        
        flops_per_time = flops / time_results.mean * 1e-12
        output, lse = forward_func(
            q_codes, q_scale, q_mn, k_codes, k_scale, k_mn, v, output_dtype=output_dtype
        )
        
        target = paddle.nn.functional.scaled_dot_product_attention(
            prepare_q.transpose([0, 2, 1, 3]),
            prepare_k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        
        loss_fn = paddle.nn.MSELoss()
        loss = loss_fn(output, target)
        
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total time {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.2f}"
            )


def benchmark_triton_attention_multi_kernel_custom(
    forward_func: Callable,
    seq_lens: list,
    num_heads: int,
    batch_size: int,
    head_dim: int,
    q_dtype: paddle.dtype = paddle.float16,
    k_dtype: paddle.dtype = paddle.float16,
    v_dtype: paddle.dtype = paddle.float16,
    output_dtype: paddle.dtype = paddle.float16,
    repeats: int = 100,
    causal: bool = False,
    logger: Optional[Any] = None,
    mixed_precision_ratio: float = 0.5,
) -> None:
    """
    Custom benchmark function for Triton attention with multi-kernel mixed precision.
    
    Args:
        forward_func (Callable): The forward function to benchmark.
        seq_lens (list): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        head_dim (int): Dimension of each attention head.
        q_dtype (paddle.dtype): Data type for query tensor.
        k_dtype (paddle.dtype): Data type for key tensor.
        v_dtype (paddle.dtype): Data type for value tensor.
        output_dtype (paddle.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        logger (Optional[Any]): Logger to log benchmarking results.
        mixed_precision_ratio (float): Ratio of int8 columns in K/V tensors (0-1).
    """
    from src.triton.utils.quant.new_pack import triton_quantize_and_pack_along_last_dim
    
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
            
        q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=q_dtype,
        )
        k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=k_dtype,
        )
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        
        k_bitmap = paddle.zeros(
            (batch_size, num_heads, seq_len), dtype=paddle.int8, device="cuda"
        )
        num_int8_cols = int(seq_len * mixed_precision_ratio)
        for b in range(batch_size):
            for h in range(num_heads):
                int8_indices = paddle.randperm(seq_len)[:num_int8_cols]
                k_bitmap[b, h, int8_indices] = 1

        def quantize_with_bitmap(data, bitmap, group_size=32, bit=8):
            quantized = paddle.zeros_like(data, dtype=paddle.int8)
            scale = paddle.zeros(
                (*data.shape[:-1], data.shape[-1] // group_size),
                dtype=paddle.float16,
                device="cuda",
            )
            for b in range(batch_size):
                for h in range(num_heads):
                    int8_cols = paddle.where(bitmap[b, h] == 1)[0]
                    if len(int8_cols) > 0:
                        col_data = data[b, h, int8_cols].unsqueeze(0).unsqueeze(0)
                        q_col, s_col, _ = triton_quantize_and_pack_along_last_dim(
                            col_data, group_size=group_size, bit=bit
                        )
                        quantized[b, h, int8_cols] = q_col
                        scale[b, h, int8_cols // group_size] = s_col
            return quantized, scale

        k_quantized, k_scale = quantize_with_bitmap(k, k_bitmap, bit=8)
        k_final = paddle.where(
            k_bitmap.unsqueeze(-1) == 1, k_quantized, k.to(paddle.int8)
        )
        q_quantized, q_scale, q_mn = triton_quantize_and_pack_along_last_dim(
            data=q, group_size=32, bit=8
        )
        
        _, time_results = benchmark_forward(
            forward_func,
            q_quantized, k_final, v, q_scale, k_scale, k_bitmap,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Mixed Precision"
        )
        
        flops_per_time = flops / time_results.mean * 1e-12
        output, lse = forward_func(
            q_quantized, k_final, v, q_scale, k_scale, k_bitmap,
            output_dtype=output_dtype,
        )
        
        target = paddle.nn.functional.scaled_dot_product_attention(
            q.transpose([0, 2, 1, 3]),
            k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        
        loss_fn = paddle.nn.MSELoss()
        try:
            loss = loss_fn(output, target)
        except:
            loss = paddle.inf
        
        if logger:
            logger.log(
                f"SeqLen {seq_len}: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.4f}, "
                f"Int8 Ratio {mixed_precision_ratio:.1%}"
            )
        else:
            print(
                f"SeqLen {seq_len}: {flops_per_time:.2f} TFLOP/s, "
                f"{time_results.mean * 1000.0:.2f} ms, "
                f"Total {time_results.mean * repeats:.2f} s, "
                f"Loss {loss:.4f}, "
                f"Int8 Ratio {mixed_precision_ratio:.1%}"
            )
