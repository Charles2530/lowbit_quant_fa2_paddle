import argparse
import datetime
import os

import paddle
import sglang
from helper import *
from inferenceKit import utils
from inferenceKit.data_config import supported_dataset
from inferenceKit.inference import inference
from inferenceKit.model_config import supported_Reward_LLM, supported_SFT_LLM
from inferenceKit.models import DefaultInferenceModel, InferenceConfig
from quant_flash_attn_triton import replace_attention_layers
from sglang.srt.distributed import (init_distributed_environment,
                                    initialize_model_parallel)


def parse_args():
    parser = argparse.ArgumentParser()
    SUPPRESS = argparse.SUPPRESS
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar="FILE",
        help="Json Config File specifying arguments",
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--reward_model", type=str, default=None)
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--cot_method", type=str, default=SUPPRESS)
    parser.add_argument("--step_method", type=str, default=SUPPRESS)
    parser.add_argument("--voting_method", type=str, default=SUPPRESS)
    parser.add_argument("--tree_width", type=int, default=SUPPRESS)
    parser.add_argument("--tree_depth", type=int, default=SUPPRESS)
    parser.add_argument("--max_new_tokens", type=int, default=SUPPRESS)
    parser.add_argument("--max_rollout", type=int, default=SUPPRESS)
    parser.add_argument("--max_step_time", type=int, default=SUPPRESS)
    parser.add_argument("--work-dir", type=str, default="./outputs")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Enable Flash Attention",
    )
    parser.add_argument(
        "--shard",
        action="store_true",
        default=False,
        help="Big Model Sharded Inference",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug Mode"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume work-dir Experiment",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args = vars(args)
>>>>>>    accelerator = accelerate.Accelerator()
    accelerator.even_batches = False
    logger = utils.get_logger("MAIN")
    device_map = "auto" if args.pop("shard") else None
    dtype = getattr(torch, args.pop("dtype"))
    flash_attn = args.pop("flash_attn")
    config_file = args.pop("config", None)
    inference_config = (
        InferenceConfig.from_json(config_file) if config_file else InferenceConfig()
    )
    utils.update_single_config(inference_config.config, copy=False, **args)
    args = inference_config.config
    generation_model, args.model_path = supported_SFT_LLM[args.model]
    generation_model = generation_model(
        model_path=args.model_path,
        device=accelerator.place,
        device_map=device_map,
        dtype=dtype,
        flash_attn=flash_attn,
    )
    reward_model = None
    if args.reward_model is not None:
        reward_model, args.reward_model_path = supported_Reward_LLM[args.reward_model]
        reward_model = reward_model(
            model_path=args.reward_model_path,
            device=accelerator.place,
            device_map=device_map,
            dtype=dtype,
            flash_attn=flash_attn,
        )
    model = DefaultInferenceModel(
        generation_model, reward_model, inference_config, device=accelerator.place
    )
    model.eval()
    replace_attention_layers(model)
    dataset = supported_dataset[args.data]()
>>>>>>    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x[0]
    )
    dataloader = accelerator.prepare(dataloader)
    final_inference_config = model.inference_config
    output_dir = None
    if accelerator.is_main_process:
        if args.resume:
            output_dir = args.resume
            resume_file = utils.get_resume_file(args.resume)
            resume_results = utils.collect_resume_results(resume_file)
        else:
            work_dir = utils.default_work_dir(args, final_inference_config)
            output_dir = utils.get_outdir(work_dir)
>>>>>>    output_dir = accelerate.utils.broadcast_object_list([output_dir], from_process=0)[0]
    if accelerator.is_main_process:
        final_inference_config.to_json(os.path.join(output_dir, "config.json"))
    results = inference(model, dataloader, accelerator, output_dir)
    for k, res in results.items():
        all_results = accelerator.gather_for_metrics(res, True)
        if accelerator.is_main_process:
            all_results = sorted(all_results, key=lambda x: x["index"])
            utils.dump(all_results, os.path.join(output_dir, f"results-{k}.json"))
            dataset.evaluate_results(all_results)


if __name__ == "__main__":
    main()
