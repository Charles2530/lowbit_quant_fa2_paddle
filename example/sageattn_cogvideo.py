import os
import sys

import paddle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from src.core import sageattn_qk_int8_pv_fp16_triton as sageattn

parser = argparse.ArgumentParser()
parser.add_argument("--compile", action="store_true", help="Compile the model")
args = parser.parse_args()
paddle.nn.functional.scaled_dot_product_attention = sageattn
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
pipe = diffusers.CogVideoXPipeline.from_pretrained(
    pretrained_model_name_or_path="./models/CogVideoX-2b", torch_dtype=paddle.float16
).to("cuda")
if args.compile:
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune-no-cudagraphs"
    )
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
device = paddle.device.get_device()
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=paddle.framework.core.default_cuda_generator(int(device[-1])).manual_seed(
        42
    ),
).frames[0]
diffusers.utils.export_to_video(video, "cogvideo_sage.mp4", fps=8)
