import sys

sys.path.append("/data/charles/codes/flash-attn-v0")
import argparse

import paddle
import sageattention._qattn as qattn
from paddle_utils import *

parser = argparse.ArgumentParser(description="Benchmark QK Int8 PV FP16 CUDA")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
parser.add_argument(
    "--quant_gran",
    type=str,
    default="per_warp",
    choices=["per_warp", "per_thread"],
    help="Quantization granularity",
)
parser.add_argument(
    "--pv_accum_dtype", type=str, default="fp16", choices=["fp16", "fp16+fp32", "fp32"]
)
args = parser.parse_args()
head = args.num_heads
batch = args.batch_size
headdim = args.head_dim
print(f"*** SAGE ATTN - CUDA QK Int8 PV FP16 ***")
print(
    f"batch: {batch}, head: {head}, headdim: {headdim}, pv_accum_dtype: {args.pv_accum_dtype}"
)
WARP_Q = 32
WARP_K = 64
if args.pv_accum_dtype == "fp32":
    kernel = qattn.qk_int8_sv_f16_accum_f32_attn
elif args.pv_accum_dtype == "fp16+fp32":
    kernel = (
        qattn.qk_int8_sv_f16_accum_f16_attn_inst_buf
        if headdim == 64
        else qattn.qk_int8_sv_f16_accum_f16_attn_buf
    )
elif args.pv_accum_dtype == "fp16":
    kernel = qattn.qk_int8_sv_f16_accum_f16_attn
_qk_quant_gran = 3 if args.quant_gran == "per_thread" else 2
is_causal = True
_is_causal = 1 if is_causal else 0
print(f"is_causal: {is_causal}")
for seq_len in {128, 512, 1024, 2048}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
    q = paddle.randint(
        low=-95, high=95, shape=(batch, seq_len, head, headdim), dtype=paddle.int8
    ).cuda()
    k = paddle.randint(
        low=-95, high=95, shape=(batch, seq_len, head, headdim), dtype=paddle.int8
    ).cuda()
    vm = paddle.randn(batch, head, headdim, dtype=paddle.float16).cuda()
    if args.quant_gran == "per_warp":
        q_scale = paddle.randn(
            batch, head, seq_len // WARP_Q, dtype=paddle.float32
        ).cuda()
        k_scale = paddle.randn(
            batch, head, seq_len // WARP_K, dtype=paddle.float32
        ).cuda()
    elif args.quant_gran == "per_thread":
        q_scale = paddle.randn(
            batch, head, seq_len // WARP_Q * 8, dtype=paddle.float32
        ).cuda()
        k_scale = paddle.randn(
            batch, head, seq_len // WARP_K * 4, dtype=paddle.float32
        ).cuda()
    v = paddle.randn(batch, seq_len, head, headdim, dtype=paddle.float16).cuda()
    o = paddle.empty(batch, seq_len, head, headdim, dtype=paddle.float16).cuda()
    sm_scale = 1 / headdim**0.5
    for i in range(5):
        kernel(q, k, v, o, q_scale, k_scale, 0, _is_causal, _qk_quant_gran, sm_scale, 0)
    paddle.device.synchronize()
>>>>>>    _, time = flash_attn.utils.benchmark.benchmark_forward(
        kernel,
        q,
        k,
        v,
        o,
        q_scale,
        k_scale,
        0,
        _is_causal,
        _qk_quant_gran,
        sm_scale,
        0,
        repeats=100,
        verbose=False,
        desc="cuda",
    )
    print(f"{seq_len} flops:{flops / time.mean * 1e-12}")
>>>>>>kernel = flash_attn.flash_attn_interface._flash_attn_forward
is_causal = True
_is_causal = 1 if is_causal else 0
window_size = -1, -1
softcap = 0.0
alibi_slopes = None
deterministic = False
return_attn_probs = False
dropout_p = 0.0
print(f"*** FLASH ATTN - CUDA QK FP16 PV FP16 ***")
print(f"is_causal: {is_causal}")
for seq_len in {128, 512, 1024, 2048}:
    flops = 4 * head * batch * headdim * seq_len * seq_len / (2 if is_causal else 1)
    q = paddle.randint(
        low=-95, high=95, shape=(batch, seq_len, head, headdim), dtype=paddle.float16
    ).cuda()
    k = paddle.randint(
        low=-95, high=95, shape=(batch, seq_len, head, headdim), dtype=paddle.float16
    ).cuda()
    vm = paddle.randn(batch, head, headdim, dtype=paddle.float16).cuda()
    v = paddle.randn(batch, seq_len, head, headdim, dtype=paddle.float16).cuda()
    o = paddle.empty(batch, seq_len, head, headdim, dtype=paddle.float16).cuda()
    sm_scale = 1 / headdim**0.5
    for i in range(5):
>>>>>>        flash_attn.flash_attn_interface._flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            sm_scale,
            is_causal,
            window_size[0],
            window_size[1],
            softcap,
            alibi_slopes,
            False,
        )
    paddle.device.synchronize()
>>>>>>    _, time = flash_attn.utils.benchmark.benchmark_forward(
        kernel,
        q,
        k,
        v,
        dropout_p,
        sm_scale,
        is_causal,
        window_size[0],
        window_size[1],
        softcap,
        alibi_slopes,
        False,
        repeats=100,
        verbose=False,
        desc="cuda",
    )
    print(f"{seq_len} flops:{flops / time.mean * 1e-12}")
