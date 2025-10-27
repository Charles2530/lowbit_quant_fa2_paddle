import os
from functools import partial

from .dataset import *

BENCHMARK_PATH = os.environ.get("BENCHMARK_ROOT", "./benchmark")
supported_dataset = {
    "simple": partial(
        SimpleDataset,
        dataset_name="simple",
        dataset_path=os.path.join(BENCHMARK_PATH, "example_dataset.json"),
    ),
    "math": partial(
        MathDataset,
        dataset_name="math",
        dataset_path=os.path.join(BENCHMARK_PATH, "math_test500_dataset.json"),
    ),
    "gsm8k": partial(
        GSM8KDataset,
        dataset_name="gsm8k",
        dataset_path=os.path.join(BENCHMARK_PATH, "gsm8k_test1319_dataset.json"),
    ),
}
