"""
author: @j-c-carr
python script to retrieve prompt datasets
"""
import torch
from typing import List, Optional, Callable, Iterable
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

### 1.1 Batching, shuffling, iteration
def build_loader(
    prompts: List[str], tokenizer, batch_size: int = 64, shuffle: bool = False, device: Optional[str] = 'cpu'
) -> Callable[[], Iterable[dict]]:
    """Custom torch loader for iterating over prompts"""

    def loader():
        """Prompt dataloader"""
        n = len(prompts)
        if shuffle:
            idxs = torch.randperm(n)
        else:
            idxs = torch.arange(0, n, dtype=torch.int)

        i = 0
        while i < n:
            yield tokenizer.batch_encode_plus([prompts[j] for j in idxs[i:i+batch_size]],
                                              padding=True, truncation=True, add_special_tokens=False,
                                               return_tensors='pt').to(device)
            i += batch_size

    return loader


def get_dataset(dset_name: str, split='train', cache_dir=None, num_samples: Optional[int] = None) -> List[str]:
    """Loads a dataset, returning prompts as a list of strings.
    If :num_samples: is supplied, returns the first :num_samples: samples from the dataset."""
    assert dset_name in ["rtp"], f'{dset_name} must be in ["rtp"]'

    if dset_name == "rtp":
        print(f'Loading RealToxicityPrompts dataset from Huggingface...')
        dataset = load_dataset('allenai/real-toxicity-prompts', split=split, cache_dir=cache_dir)
        dataset = dataset['prompt'][:num_samples]
        dataset = [d['text'] for d in dataset]

    return dataset

def add_instruction_format(prompts: List[str]) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    return [f'\n\nHuman: {prompt}\n\nAssistant:' for prompt in prompts]


def get_prompt_dataloader(dset_name: str, tokenizer, add_instruction_prefix=False, shuffle=False, **get_dataset_kwargs):
    """Loads data"""
    prompts = get_dataset(dset_name, **get_dataset_kwargs)

    if add_instruction_prefix:
        prompts = add_instruction_format(prompts)

    # Encode the prompts
    print('Processing dataset')

    dataloader = build_loader(prompts, tokenizer)

    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataloader = get_prompt_dataloader("rtp", tokenizer, num_samples=1000)

    for batch in dataloader():
        print(batch)
        exit(0)
    print('Done.')