import sys

sys.path.append("/data/charles/codes/flash-attn-v0")
import paddle
from paddle_utils import *

"""
modified from: https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_example.py
sh ./run_parallel.sh
"""
import time
from functools import partial

import src
from xfuser import xFuserArgs, xFuserCogVideoXPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (get_runtime_state, get_world_group,
                                     is_dp_last_group)

paddle.set_grad_enabled(False)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument(
        "--use_sage_attn_fp16",
        action="store_true",
        help="Use Sage Attention fp16 or not.",
    )
    parser.add_argument(
        "--use_sage_attn_fp8",
        action="store_true",
        help="Use Sage Attention fp8 or not.",
    )
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    num_heads = 30
    if engine_args.ulysses_degree > 0 and num_heads % engine_args.ulysses_degree != 0:
        raise ValueError(
            f"ulysses_degree ({engine_args.ulysses_degree}) must be a divisor of the number of heads ({num_heads})"
        )
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    sage_tag = "sage+None"
    if args.use_sage_attn_fp16:
        paddle.nn.functional.scaled_dot_product_attention = partial(
            src.sageattn_qk_int8_pv_fp16_cuda, pv_accum_dtype="fp32"
        )
        sage_tag = f"sage+fp16"
    elif args.use_sage_attn_fp8:
        paddle.nn.functional.scaled_dot_product_attention = partial(
            src.sageattn_qk_int8_pv_fp8_cuda, pv_accum_dtype="fp32+fp32"
        )
        sage_tag = f"sage+fp8"
    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=paddle.bfloat16,
    )
    if args.enable_sequential_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
    else:
        device = device2str(f"cuda:{local_rank}")
        pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    paddle.device.cuda.reset_max_memory_allocated()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=paddle.framework.core.default_cpu_generator().manual_seed(
            input_config.seed
        ),
        guidance_scale=6,
        use_dynamic_cfg=True,
        latents=None,
    ).frames[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = paddle.device.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    parallel_info = f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_tp{engine_args.tensor_parallel_degree}_pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_compile{engine_config.runtime_config.use_torch_compile}"
    if is_dp_last_group():
        world_size = get_world_group().world_size
        prompt_tag: str = input_config.prompt[0]
        prompt_tag = prompt_tag.replace(" ", "_").replace(".", "")
        resolution = (
            f"{input_config.width}x{input_config.height}x{input_config.num_frames}"
        )
        output_filename = f"results/cogvideox_{parallel_info}_{sage_tag}_{world_size}gpu_{resolution}_{prompt_tag}.mp4"
>>>>>>        diffusers.utils.export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")
    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory / 1000000000.0} GB"
        )
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
