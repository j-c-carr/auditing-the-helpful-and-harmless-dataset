"""
author: @j-c-carr
python script to retrieve prompt datasets
"""
import torch
from collections import defaultdict
from typing import List, Optional
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_prompts(dset_name: str, split='train', cache_dir=None, data_dir=None, num_samples: Optional[int] = None) -> List[str]:
    """Loads a dataset, returning prompts as a list of strings.
    If :num_samples: is supplied, returns the first :num_samples: samples from the dataset."""
    assert dset_name in ["rtp", "hh", "fairprism", "xstest"], f'{dset_name} must be in ["rtp", "hh", "fairprism", "xstest"]'

    if dset_name == "rtp":
        print(f'Loading RealToxicityPrompts dataset from Huggingface...')
        dataset = load_dataset('allenai/real-toxicity-prompts', split=split, cache_dir=cache_dir)
        prompts = [prompt['text'] for prompt in dataset['prompt']]

    elif dset_name == "fairprism":
        # Assumes that fairprism_aggregated.csv is in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/fairprism_aggregated.csv')['Human Input'].tolist()

    elif dset_name == "xstest":
        # Assumes that xstest prompts are in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/xstest_v2_prompts.csv')['prompt'].tolist()

    elif dset_name == "hh":
        print(f'Loading HH dataset ({split} split) from Huggingface...')
        dataset = load_dataset('Anthropic/hh-rlhf', data_dir=data_dir, split=split, cache_dir=cache_dir)
        print('done')

        # Source: https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py
        def _extract_anthropic_prompt(prompt_and_response):
            """Extract the anthropic prompt from a prompt and response pair.
            Taken from DPO: https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py
            """
            search_term = '\n\nAssistant:'
            search_term_idx = prompt_and_response.rfind(search_term)
            assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
            return prompt_and_response[:search_term_idx + len(search_term)]

        from tqdm import tqdm
        prompts = []
        for row in tqdm(dataset, desc='Processing HH'):
            prompt = _extract_anthropic_prompt(row['chosen'])
            prompts.append(prompt)

    return prompts[:num_samples]

def add_instruction_format(prompts: List[str]) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    return [f'\n\nHuman: Continue the following sentence: {prompt}\n\nAssistant:' for prompt in prompts]


if __name__ == '__main__':
    prompts = get_prompts("xstest", cache_dir='data/xstest')
    dataloader = DataLoader(prompts, shuffle=False, batch_size=64)
