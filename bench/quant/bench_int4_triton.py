import os
import sys

import paddle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.triton.attn_qk_int8_per_block_causal import forward as forward_causal
from src.triton.quantization.attn_4bit_per_block import \
    _quantized_flash_attn_forward as forward
from utils.benchmark import benchmark_triton_attention
from utils.logger_util import Logger, eval_log
from utils.parser_util import get_args, get_save_name


def pipeline(args, logger):
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_lens = [128, 1024, 2048, 4096, 8192]
    assert args.method in ["fa2", "torch", "xformers"]
>>>>>>    torch.backends.cuda.enable_flash_sdp(args.method == "fa2")
>>>>>>    torch.backends.cuda.enable_math_sdp(args.method == "torch")
>>>>>>    torch.backends.cuda.enable_mem_efficient_sdp(args.method == "xformers")
    logger.log(f"Triton Int4 : {args.method}")
    logger.log(
        f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}"
    )
    logger.log(f"Non-causal Attention benchmark")
    benchmark_triton_attention(
        forward_func=forward,
        seq_lens=seq_lens,
        num_heads=num_heads,
        batch_size=batch_size,
        head_dim=head_dim,
        causal=False,
        logger=logger,
        bits=4,
    )
    logger.log(f"Causal Attention benchmark")
    benchmark_triton_attention(
        forward_func=forward_causal,
        seq_lens=seq_lens,
        num_heads=num_heads,
        batch_size=batch_size,
        head_dim=head_dim,
        causal=True,
        logger=logger,
        bits=4,
    )


def main():
    args = get_args(desc="Benchmark QK Int8 PV FP16 Triton")
    logger = Logger(
        name="triton_flash_attn_qk_int8_pv_fp16_" + get_save_name(args),
        log_dir="logs",
        logging_mode=True,
    )
    eval_func = eval_log(logger)(lambda: pipeline(args, logger))
    eval_func()


if __name__ == "__main__":
    main()
