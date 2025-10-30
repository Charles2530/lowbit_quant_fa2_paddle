import sys

sys.path.append("/data/charles/codes/flash-attn-v0")
import paddle
from paddle_utils import *
from src.triton.quant_per_block import (per_block_int4, per_block_int8,
                                        per_block_q_int8_k_int4)
from src.triton.quant_per_thread import per_thread_int4, per_thread_int8
from src.triton.utils.quant.new_pack import \
    triton_quantize_and_pack_along_last_dim
from utils.utils import unpack_and_dequant_ocache
from utils.paddle_package import benchmark_forward

def benchmark_attention(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark scaled dot-product attention (forward_func) performance with varying sequence lengths.

    Args:
        forward_func (callable): Function implementing scaled dot-product attention.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        seq_len_list (list): List of sequence lengths to benchmark.
        logger (object): Logger to log benchmarking results.
        is_causal (bool, optional): Whether attention is causal. Defaults to False.
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
        for _ in range(5):
            forward_func(q, k, v, is_causal=causal)
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = paddle.flash_attn.utils.benchmark.benchmark_forward(
            forward_func,
            q,
            k,
            v,
            is_causal=causal,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        if logger:
            logger.log(
                f"{seq_len} flops: {flops / time.mean * 1e-12:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s"
            )
        else:
            print(
                f"{seq_len} flops: {flops / time.mean * 1e-12:.2f} TFLOP/s",
                f"{time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s",
            )


def benchmark_triton_attention_int8_baseline(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    q_scale_shape_divisor=128,
    k_scale_shape_divisor=64,
    output_dtype=paddle.bfloat16,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        q_scale = paddle.randn(
            batch_size,
            num_heads,
            seq_len // q_scale_shape_divisor,
            1,
            dtype=v_dtype,
            device="cuda",
        )
        k_scale = paddle.randn(
            batch_size,
            num_heads,
            seq_len // k_scale_shape_divisor,
            1,
            dtype=v_dtype,
            device="cuda",
        )
        for _ in range(5):
            forward_func(q, k, v, q_scale, k_scale, output_dtype=output_dtype)
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q,
            k,
            v,
            q_scale,
            k_scale,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean * 1e-12
        output, _ = forward_func(q, k, v, q_scale, k_scale, output_dtype=output_dtype)
        target = paddle.nn.functional.scaled_dot_product_attention(
            q.transpose([0, 2, 1, 3]),
            k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        loss_fn = paddle.nn.MSELoss()
        loss = loss_fn(output, target)
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention_int8(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.bfloat16,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
        v = paddle.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=v_dtype
        )
        v = v.cuda()
        quant_type = 1
        if quant_type == 0:
            q_codes, q_scale, _ = triton_quantize_and_pack_along_last_dim(
                data=prepare_q, group_size=32, bit=8
            )
            k_codes, k_scale, _ = triton_quantize_and_pack_along_last_dim(
                data=prepare_k, group_size=32, bit=8
            )
        elif quant_type == 1:
            q_codes, q_scale, k_codes, k_scale = per_block_int8(prepare_q, prepare_k)
        else:
            q_codes, q_scale, k_codes, k_scale = per_thread_int8(prepare_q, prepare_k)
        for _ in range(5):
            forward_func(
                q_codes, k_codes, v, q_scale, k_scale, output_dtype=output_dtype
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q_codes,
            k_codes,
            v,
            q_scale,
            k_scale,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean() * 1e-12
        q_output, _ = forward_func(
            q_codes, k_codes, v, q_scale, k_scale, output_dtype=output_dtype
        )
        # import pdb; pdb.set_trace()
        target = paddle.nn.functional.scaled_dot_product_attention(
            prepare_q,
            prepare_k,
            v,
            is_causal=causal,
        )
        loss_fn = paddle.nn.MSELoss()
        loss = loss_fn(q_output, target)
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean() * 1000.0:.2f} ms,Total time {time.mean() * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean() * 1000.0:.2f} ms,Total time {time.mean() * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention_int4(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.float16,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        quant_type = 1
        if quant_type == 0:
            q_codes, q_scale, q_mn = triton_quantize_and_pack_along_last_dim(
                data=prepare_q, group_size=32, bit=4
            )
            k_codes, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
                data=prepare_k, group_size=32, bit=4
            )
        elif quant_type == 1:
            q_codes, q_scale, k_codes, k_scale = per_block_int4(prepare_q, prepare_k)
        else:
            q_codes, q_scale, k_codes, k_scale = per_thread_int4(prepare_q, prepare_k)
        for _ in range(5):
            forward_func(
                q_codes, q_scale, k_codes, k_scale, v, output_dtype=output_dtype
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q_codes,
            q_scale,
            k_codes,
            k_scale,
            v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean * 1e-12
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
            import pdb

            pdb.set_trace()
            pass
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention_q_int8_k_int4(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.float16,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        quant_type = 1
        if quant_type == 0:
            q_codes, q_scale, q_mn = triton_quantize_and_pack_along_last_dim(
                data=prepare_q, group_size=32, bit=4
            )
            k_codes, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
                data=prepare_k, group_size=32, bit=4
            )
        elif quant_type == 1:
            q_codes, q_scale, k_codes, k_scale = per_block_q_int8_k_int4(
                prepare_q, prepare_k
            )
        else:
            q_codes, q_scale, k_codes, k_scale = per_thread_int4(prepare_q, prepare_k)
        for _ in range(5):
            forward_func(
                q_codes, q_scale, k_codes, k_scale, v, output_dtype=output_dtype
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q_codes,
            q_scale,
            k_codes,
            k_scale,
            v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean * 1e-12
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
            import pdb

            pdb.set_trace()
            pass
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention_int2(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.float16,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        prepare_q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        prepare_k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
        v = paddle.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=v_dtype, device="cuda"
        )
        q_codes, q_scale, q_mn = triton_quantize_and_pack_along_last_dim(
            prepare_q, group_size=32, bit=8
        )
        k_codes, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
            prepare_k, group_size=32, bit=2
        )
        for _ in range(5):
            forward_func(
                q_codes,
                q_scale,
                q_mn,
                k_codes,
                k_scale,
                k_mn,
                v,
                output_dtype=output_dtype,
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q_codes,
            q_scale,
            q_mn,
            k_codes,
            k_scale,
            k_mn,
            v,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean * 1e-12
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
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.bfloat16,
    bits=8,
    repeats=100,
    causal=False,
    logger=None,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        q_scale_shape_divisor (int): Divisor to calculate query scale shape.
        k_scale_shape_divisor (int): Divisor to calculate key scale shape.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        query_states = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, seq_len, num_heads, head_dim),
            dtype=q_dtype,
        )
        key_states = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, seq_len, num_heads, head_dim),
            dtype=k_dtype,
        )
        value_states = paddle.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=v_dtype, device="cuda"
        )
        prepare_k = key_states.transpose(1, 3)
        k_codes, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(
            data=prepare_k, group_size=32, bit=bits
        )
        v_codes, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(
            data=value_states, group_size=32, bit=bits
        )
        softmax_scale = query_states.shape[-1] ** -0.5
        for _ in range(5):
            forward_func(
                query_states,
                k_codes,
                k_scale,
                k_mn,
                v_codes,
                v_scale,
                v_mn,
                group_size=32,
                bits=bits,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            query_states,
            k_codes,
            k_scale,
            k_mn,
            v_codes,
            v_scale,
            v_mn,
            group_size=32,
            bits=bits,
            softmax_scale=softmax_scale,
            causal=causal,
            repeats=repeats,
            verbose=False,
            desc="Triton",
        )
        flops_per_time = flops / time.mean * 1e-12
        q_output, _, _ = forward_func(
            query_states,
            k_codes,
            k_scale,
            k_mn,
            v_codes,
            v_scale,
            v_mn,
            group_size=32,
            bits=bits,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        target = paddle.nn.functional.scaled_dot_product_attention(
            query_states.transpose([0, 2, 1, 3]),
            key_states.transpose([0, 2, 1, 3]),
            value_states.transpose([0, 2, 1, 3]),
            is_causal=causal,
        ).transpose([0, 2, 1, 3])
        loss_fn = paddle.nn.MSELoss()
        loss = loss_fn(q_output, target)
        if logger:
            logger.log(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )
        else:
            print(
                f"{seq_len} flops: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms,Total time {time.mean * repeats:.2f} s,Loss {loss:.2f}"
            )


def benchmark_triton_attention_multi_kernel(
    forward_func,
    seq_lens,
    num_heads,
    batch_size,
    head_dim,
    q_dtype=paddle.float16,
    k_dtype=paddle.float16,
    v_dtype=paddle.float16,
    output_dtype=paddle.float16,
    repeats=100,
    causal=False,
    logger=None,
    mixed_precision_ratio=0.5,
):
    """
    Benchmark attention forward functions with different sequence lengths.

    Args:
        forward_func (callable): The forward function to benchmark.
        seq_lens (list or set): Sequence lengths to test.
        num_heads (int): Number of attention heads.
        batch_size (int): batch_size size.
        head_dim (int): Dimension of each attention head.
        q_dtype (torch.dtype): Data type for query tensor.
        k_dtype (torch.dtype): Data type for key tensor.
        v_dtype (torch.dtype): Data type for value tensor.
        output_dtype (torch.dtype): Data type for the output.
        repeats (int): Number of repetitions for benchmarking.
        causal (bool): Whether the forward function is causal.
        mixed_precision_ratio (float): Ratio of int8 columns in K/V tensors (0-1).
    """
    for seq_len in seq_lens:
        flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
        if causal:
            flops //= 2
        q = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(q_dtype)
        k = paddle.randint(
            low=-100,
            high=100,
            shape=(batch_size, num_heads, seq_len, head_dim),
            dtype=paddle.int32,
        ).astype(k_dtype)
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
        for _ in range(5):
            forward_func(
                q_quantized,
                k_final,
                v,
                q_scale,
                k_scale,
                k_bitmap,
                output_dtype=output_dtype,
            )
        paddle.device.synchronize()
# >>>>>>        _, time = flash_attn.utils.benchmark.benchmark_forward(
        _, time = benchmark_forward(
            forward_func,
            q_quantized,
            k_final,
            v,
            q_scale,
            k_scale,
            k_bitmap,
            output_dtype=output_dtype,
            repeats=repeats,
            verbose=False,
            desc="Triton Mixed Precision",
        )
        flops_per_time = flops / time.mean * 1e-12
        output, lse = forward_func(
            q_quantized,
            k_final,
            v,
            q_scale,
            k_scale,
            k_bitmap,
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
            import pdb

            pdb.set_trace()
        if logger:
            logger.log(
                f"SeqLen {seq_len}: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms, Total {time.mean * repeats:.2f} s, Loss {loss:.4f}, Int8 Ratio {mixed_precision_ratio:.1%}"
            )
        else:
            print(
                f"SeqLen {seq_len}: {flops_per_time:.2f} TFLOP/s, {time.mean * 1000.0:.2f} ms, Total {time.mean * repeats:.2f} s, Loss {loss:.4f}, Int8 Ratio {mixed_precision_ratio:.1%}"
            )
