"""
author: @j-c-carr
python script to retrieve prompt datasets
"""
import torch
from collections import defaultdict
from typing import List, Optional, Tuple
from datasets import load_dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader


def extract_hh_prompt_from_sample(prompt_and_response: str) -> str:
    """Extract the prompt from a prompt-response pair from the HH dataset."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def extract_hh_prompts(samples):
    """Get the prompts from the dataset"""
    prompt = [extract_hh_prompt_from_sample(chosen_str) for chosen_str in samples['chosen']]
    return {'prompt': prompt}


def get_hh_prompts(data_dirs: List[str], split: str, cache_dir: str = None) -> List[str]:
    """Load the prompts from the Anthropic Helpful-Harmless dataset.

       Prompts are be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    assert len(data_dirs) > 0, 'Must provide at least one data directory'

    print(f'Loading HH ({split} split) from {data_dirs}...')
    datasets = []
    for data_dir in data_dirs:
        dataset = load_dataset('Anthropic/hh-rlhf', split=split, data_dir=data_dir, cache_dir=cache_dir)
        datasets.append(dataset)

    dataset = concatenate_datasets(datasets)
    print('done')

    # Preprocess the dataset to prepare for TRL training
    original_columns = dataset.column_names
    dataset = dataset.map(extract_hh_prompts, batched=True,remove_columns=original_columns)

    return dataset['prompt']


def get_prompts(dset_name: str, split='train', cache_dir=None, data_dir=None, num_samples: Optional[int] = None) -> List[str]:
    """Loads a dataset, returning prompts as a list of strings.
    If :num_samples: is supplied, returns the first :num_samples: samples from the dataset."""
    assert dset_name in ["rtp", "hh",  "hh-helpful-only",  "hh-harmless-only", "fairprism", "xstest"], f'{dset_name} must be in ["rtp", "hh",  "hh-helpful-only", "hh-harmless-only", "fairprism", "xstest"]'

    if dset_name == "rtp":
        print(f'Loading RealToxicityPrompts dataset from Huggingface...')
        dataset = load_dataset('allenai/real-toxicity-prompts', split=split, cache_dir=cache_dir)
        prompts = [prompt['text'] for prompt in dataset['prompt']]

    elif dset_name == "fairprism":
        # Assumes that fairprism_aggregated.csv is in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/fairprism_aggregated.csv')['Human Input'].tolist()
        prompts = list(set(prompts))    # Remove duplicate prompts

    elif dset_name == "xstest":
        # Assumes that xstest prompts are in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/xstest_v2_prompts.csv')['prompt'].tolist()

    elif dset_name == "hh":
        data_dirs = ["helpful-base", "harmless-base", "helpful-online", "helpful-rejection-sampled"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    elif dset_name == "hh-helpful-only":
        data_dirs = ["helpful-online", "helpful-base", "helpful-rejection-sampled"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    elif dset_name == "hh-harmless-only":
        data_dirs = ["harmless-base"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    else:
        raise ValueError

    return prompts[:num_samples]

def get_prompts_from_csv(filename: str, prompt_col_name: str, num_samples: Optional[int] = None) -> List[str]:
    """Loads prompts from a csv file"""
    assert filename.endswith(".csv"), 'filename must end with .csv'
    import pandas as pd

    prompts = pd.read_csv(filename)[prompt_col_name].tolist()

    return [p[:1024] for p in prompts[:num_samples]]

def add_instruction_format(prompts: List[str], dset_name=None) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    if dset_name == 'xstest':
        return [f'\n\nHuman: {prompt}\n\nAssistant:' for prompt in prompts]
    else:
        return [f'\n\nHuman: Hi! Could you help me finish a sentence? The sentence is: {prompt}\n\nAssistant:' for prompt in prompts]


if __name__ == '__main__':
    prompts = get_prompts("xstest", cache_dir='data/xstest')
    dataloader = DataLoader(prompts, shuffle=False, batch_size=64)
