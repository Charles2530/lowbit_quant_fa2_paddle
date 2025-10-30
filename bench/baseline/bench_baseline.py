import os
import sys

import paddle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.benchmark import benchmark_attention
from utils.logger_util import Logger, eval_log
from utils.parser_util import get_args, get_save_name
from paddle.nn.functional import scaled_dot_product_attention as sdpa


def pipeline(args, logger):
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    assert args.method in ["fa2", "torch", "xformers"]
# >>>>>>    torch.backends.cuda.enable_flash_sdp(args.method == "fa2")
# >>>>>>    torch.backends.cuda.enable_math_sdp(args.method == "torch")
# >>>>>>    torch.backends.cuda.enable_mem_efficient_sdp(args.method == "xformers")
    logger.log(f"Baseline: {args.method}")
    logger.log(
        f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}"
    )
    logger.log(f"Non-causal Attention benchmark")
    benchmark_attention(
        forward_func=sdpa,
        seq_lens=seq_lens,
        num_heads=num_heads,
        batch_size=batch_size,
        head_dim=head_dim,
        causal=False,
        logger=logger,
    )
    logger.log(f"Causal Attention benchmark")
    benchmark_attention(
        forward_func=sdpa,
        seq_lens=seq_lens,
        num_heads=num_heads,
        batch_size=batch_size,
        head_dim=head_dim,
        causal=True,
        logger=logger,
    )


def main():
    args = get_args(desc="Benchmark Baseline")
    logger = Logger(
        name="baseline_flash_attn_" + get_save_name(args),
        log_dir="logs",
        logging_mode=True,
    )
    eval_func = eval_log(logger)(lambda: pipeline(args, logger))
    eval_func()


if __name__ == "__main__":
    main()
