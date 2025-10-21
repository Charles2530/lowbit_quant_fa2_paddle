import os
import sys

import t2v_metrics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *

vqa_score_metric = t2v_metrics.VQAScore(model="clip-flant5-xxl")
clip_score_metric = t2v_metrics.CLIPScore(model="openai:ViT-L-14-336")
itm_score_metric = t2v_metrics.ITMScore(model="blip2-itm")
result_path = "/data/charles/codes/flash-attn-triton/video"
image_path = [
    result_path + "/cogvideo_sage_baseline/frame_000000.jpg",
    result_path + "/cogvideo_sage_int4/frame_000000.jpg",
    result_path + "/cogvideo_sage_int8/frame_000000.jpg",
    result_path + "/cogvideo_sage_multi/frame_000000.jpg",
]
text = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
metric = {}
for img in image_path:
    vqa_score = vqa_score_metric(images=[img], texts=[text])
    clip_score = clip_score_metric(images=[img], texts=[text])
    itms_score = itm_score_metric(images=[img], texts=[text])
    name = img.split("/")[-2]
    metric[name] = {
        "vqa_score": vqa_score.item(),
        "clip_score": clip_score.item(),
        "itms_score": itms_score.item(),
    }
import json

with open("result.json", "w") as f:
    json.dump(metric, f, indent=4)
print(metric)
