import glob
import os
import sys
import traceback
from typing import Dict, List, Union

import paddle
from inferenceKit import utils
from tqdm import tqdm

from .models import BaseInferenceModel
from .utils import InferenceConfig


def inference(
    model: BaseInferenceModel,
    dataloader: torch.utils.data.DataLoader,
    accelerator: accelerate.Accelerator,
    output_dir: str,
    **kwargs
) -> List[Dict]:
    prev_tmpl = os.path.join(output_dir, "{}_{}_{}_PREV.json")
    ind, num = accelerator.process_index, accelerator.num_processes
    results = {}
    try:
        for i, data in enumerate(tqdm(dataloader)):
            input = data["input"]
            response_dict = model.generate(input, **kwargs)
            res = data
            for k, response in response_dict.items():
                if k not in results:
                    results[k] = []
                res["response"] = response
                results[k].append(res)
            for k in results.keys():
                utils.dump(results[k], prev_tmpl.format(ind, num, k))
    except (Exception, KeyboardInterrupt) as e:
        for k, res in results.items():
            all_results = accelerator.gather_for_metrics(res, True)
            if accelerator.is_main_process:
                all_results = sorted(all_results, key=lambda x: x["index"])
                resume_tmpl = os.path.join(output_dir, "RESUME_{}_{}.json")
                i = 0
                while os.path.exists(resume_tmpl.format(k, i)):
                    i += 1
                utils.dump(all_results, resume_tmpl.format(k, i))
                for file in glob.iglob(prev_tmpl.format("*", "*", k)):
                    os.remove(file)
        traceback.print_exc()
        sys.exit()
    return results
