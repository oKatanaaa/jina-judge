from copy import deepcopy
from random import shuffle
import datasets
import torch
import glob
import json
import os


def load_dataset(path) -> dict:
    dataset = {}
    for model in glob.glob(os.path.join(path, "*.jsonl")):
        
        with open(model, "r") as f:
            data = [json.loads(l) for l in f.readlines()]
        model_name = os.path.basename(model)
        model_name = model_name.replace(".jsonl", "")
        dataset[model_name] = data
    return dataset


prompt_template = """
<user prompt>
{user_prompt}
<end>
<assistant A answer>
{assistant_a}
<end>
<assistant B answer>
{assistant_b}
<end>
""".strip()


class DataGenerator:
    def __init__(self, prompt_template, tokenizer, data, max_tokens=4096):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.data = data
        self.max_tokens = max_tokens

    def format(self, item):
        user_prompt = item["user_prompt"]
        assistant_a = item["assistant_a_answer"]
        assistant_b = item["assistant_b_answer"]
        prompt = self.prompt_template.format(
            user_prompt=user_prompt,
            assistant_a=assistant_a,
            assistant_b=assistant_b,
        )
        return prompt
    
    def __iter__(self):
        data = deepcopy(self.data)
        shuffle(data)
        for item in data:
            prompt = self.format(item)
            if len(self.tokenizer.encode(prompt)) > self.max_tokens:
                continue
            #tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens, padding='max_length')
            #tokens['input_ids'] = tokens['input_ids'][0]
            #tokens['attention_mask'] = tokens['attention_mask'][0]
            tokens = {}
            tokens['prompt'] = prompt
            tokens['score'] = item["score"]
            yield tokens

    def __call__(self):
        return self.__iter__()
    

def make_dataloader(data, tokenizer, batch_size=64):
    set_list = []
    for v in data.values():
        set_list.extend(v)
    gen = DataGenerator(prompt_template, tokenizer, set_list)
    dataset = datasets.IterableDataset.from_generator(gen)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader