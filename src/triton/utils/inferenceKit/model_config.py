from functools import partial

from . import *
from .models import *

supported_SFT_LLM = {
    "llama3-8b": (DefaultLargeLanguageModel, "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen-1.5b": (QwenLLM, "Qwen/Qwen2.5-Math-1.5B-Instruct"),
    "qwen-7b": (QwenLLM, "Qwen/Qwen2.5-Math-7B-Instruct"),
    "qwen-1.5b-slang": (QwenSlang, "Qwen/Qwen2.5-Math-7B-Instruct"),
    "qwen-1.5b-vllm": (QwenLLM_vLLM, "Qwen/Qwen2.5-Math-1.5B-Instruct"),
}
supported_Reward_LLM = {
    "mistral_prm-7b": (MistralPRM, "peiyi9979/math-shepherd-mistral-7b-prm")
}
