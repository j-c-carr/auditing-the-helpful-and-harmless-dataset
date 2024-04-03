"""
author: @j-c-carr
python script to retrieve prompt datasets
"""
import torch
from typing import List, Optional, Callable, Iterable
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def get_prompts(dset_name: str, split='train', cache_dir=None, num_samples: Optional[int] = None) -> List[str]:
    """Loads a dataset, returning prompts as a list of strings.
    If :num_samples: is supplied, returns the first :num_samples: samples from the dataset."""
    assert dset_name in ["rtp"], f'{dset_name} must be in ["rtp"]'

    if dset_name == "rtp":
        print(f'Loading RealToxicityPrompts dataset from Huggingface...')
        dataset = load_dataset('allenai/real-toxicity-prompts', split=split, cache_dir=cache_dir)
        prompts = [prompt['text'] for prompt in dataset['prompt'][:num_samples]]    # TODO: shuffle examples

    return prompts

def add_instruction_format(prompts: List[str]) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    return [f'\n\nHuman: Continue the following sentence: {prompt}\n\nAssistant:' for prompt in prompts]


if __name__ == '__main__':
    prompts = get_prompts("rtp", num_samples=1000)
    dataloader = DataLoader(prompts, shuffle=False, batch_size=64)
