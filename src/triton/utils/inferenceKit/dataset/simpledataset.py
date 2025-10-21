from ..utils import load
from .basedataset import BaseDataset


class SimpleDataset(BaseDataset):
    def load_data(self, dataset_path, **kwargs):
        dataset = load(dataset_path)
        data_list = []
        for idx, data in enumerate(dataset):
            if "index" not in data:
                raise ValueError(f"Not found index in dataset {dataset_path} !!!")
            index = data["index"]
            prompt = data["input"]
            query = self.construct_instruction(prompt)
            answer = data["label"]
            dic = {"index": index, "input": query, "answer": answer}
            data_list.append(dic)
        return data_list

    def construct_instruction(self, prompt):
        return prompt
