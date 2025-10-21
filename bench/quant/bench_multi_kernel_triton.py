import os
import sys

import paddle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.triton.quantization.attn_multi_kernel_per_block import \
    forward as forward
from src.triton.quantization.attn_multi_kernel_per_block import \
    forward as forward_causal
from utils.benchmark import benchmark_triton_attention_multi_kernel
from utils.logger_util import Logger, eval_log
from utils.parser_util import get_args, get_save_name


def pipeline(args, logger):
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_lens = [1024, 2048, 4096, 8192]
    mixed_precision_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
    assert args.method in ["fa2", "torch", "xformers"]
>>>>>>    torch.backends.cuda.enable_flash_sdp(args.method == "fa2")
>>>>>>    torch.backends.cuda.enable_math_sdp(args.method == "torch")
>>>>>>    torch.backends.cuda.enable_mem_efficient_sdp(args.method == "xformers")
    logger.log(f"Benchmark Method: {args.method}")
    logger.log(
        f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}"
    )
    logger.log(f"Testing sequence lengths: {seq_lens}")
    logger.log(f"Testing mixed precision ratios: {mixed_precision_ratios}")
    for ratio in mixed_precision_ratios:
        logger.log(f"=== Mixed Precision Ratio: {ratio:.0%} ===")
        logger.log(f"Non-causal Attention benchmark (Int8 ratio: {ratio:.0%})")
        benchmark_triton_attention_multi_kernel(
            forward_func=forward,
            seq_lens=seq_lens,
            num_heads=num_heads,
            batch_size=batch_size,
            head_dim=head_dim,
            causal=False,
            logger=logger,
            mixed_precision_ratio=ratio,
        )
        logger.log(f"Causal Attention benchmark (Int8 ratio: {ratio:.0%})")
        benchmark_triton_attention_multi_kernel(
            forward_func=forward_causal,
            seq_lens=seq_lens,
            num_heads=num_heads,
            batch_size=batch_size,
            head_dim=head_dim,
            causal=True,
            logger=logger,
            mixed_precision_ratio=ratio,
        )


def main():
    args = get_args(desc="Benchmark Multi Precision Triton")
    logger = Logger(
        name="triton_flash_attn_multi_kernel_" + get_save_name(args),
        log_dir="logs",
        logging_mode=True,
    )
    eval_func = eval_log(logger)(lambda: pipeline(args, logger))
    eval_func()


if __name__ == "__main__":
    main()
