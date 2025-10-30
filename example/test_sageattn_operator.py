import argparse
import os
import sys
import time

import paddle

try:
    from src import (
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_qk_int4_pv_fp16_triton,
    )
except ModuleNotFoundError:
    # Allow running without installing the package
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    from src import (
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_qk_int4_pv_fp16_triton,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick test for low-bit FlashAttention Paddle operators",
    )
    parser.add_argument("--op", type=str, default="int8",
                        choices=["int8", "int4"],
                        help="Which quantized operator to test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--layout", type=str, default="HND",
                        choices=["HND", "NHD"],
                        help="Tensor layout")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def make_inputs(batch_size, num_heads, seq_len, head_dim, layout):
    if layout == "HND":
        q = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)
        k = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)
        v = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)
    else:
        q = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)
        k = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)
        v = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)
    return q.cuda(), k.cuda(), v.cuda()


def reference_sdpa(q, k, v, layout, causal):
    if layout == "HND":
        q_ref = q.transpose([0, 2, 1, 3])
        k_ref = k.transpose([0, 2, 1, 3])
        v_ref = v.transpose([0, 2, 1, 3])
        out = paddle.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
        return out.transpose([0, 2, 1, 3])
    else:
        return paddle.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def run_once(op, q, k, v, layout, causal):
    if op == "int8":
        out = sageattn_qk_int8_pv_fp16_triton(q=q, k=k, v=v, tensor_layout=layout, is_causal=causal)
    else:
        out = sageattn_qk_int4_pv_fp16_triton(q=q, k=k, v=v, tensor_layout=layout, is_causal=causal)
    return out


def main():
    args = parse_args()
    assert paddle.is_compiled_with_cuda(), "CUDA build of PaddlePaddle is required."

    q, k, v = make_inputs(args.batch_size, args.num_heads, args.seq_len, args.head_dim, args.layout)

    # Warmup
    for _ in range(10):
        _ = run_once(args.op, q, k, v, args.layout, args.causal)
    paddle.device.synchronize()

    # Timing
    t0 = time.time()
    for _ in range(args.repeats):
        out = run_once(args.op, q, k, v, args.layout, args.causal)
    paddle.device.synchronize()
    t1 = time.time()

    # Correctness (MSE vs. FP reference)
    ref = reference_sdpa(q, k, v, args.layout, args.causal)
    mse = paddle.nn.functional.mse_loss(out.astype(ref.dtype), ref).item()

    num_flops = 4 * args.batch_size * args.num_heads * args.head_dim * args.seq_len * args.seq_len
    if args.causal:
        num_flops //= 2
    avg_ms = (t1 - t0) * 1000.0 / args.repeats
    tflops = (num_flops / ((t1 - t0) / args.repeats)) * 1e-12

    print(f"Operator: {args.op}, layout: {args.layout}, causal: {args.causal}")
    print(f"Shape: B{args.batch_size} H{args.num_heads} N{args.seq_len} D{args.head_dim}")
    print(f"Latency: {avg_ms:.2f} ms, Throughput: {tflops:.2f} TFLOP/s, MSE: {mse:.4e}")


if __name__ == "__main__":
    main()


