"""
Utility script for extracting helpful prompts with identity terms
"""
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets

def extract_hh_prompt_from_sample(prompt_and_response: str) -> str:
    """Extract the prompt from a prompt-response pair from the HH dataset."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(samples):
    """Preprocesses data"""
    prompt = [extract_hh_prompt_from_sample(chosen_str) for chosen_str in samples['chosen']]
    chosen_response = [chosen_str[len(prompt[i]):] for i, chosen_str in enumerate(samples['chosen'])]
    rejected_response = [rejected_str[len(prompt[i]):] for i, rejected_str in enumerate(samples['rejected'])]
    return {'prompt': prompt, 'chosen': chosen_response, 'rejected': rejected_response}

def truncate_prompt(prompt: str, term: str) -> str:
    """In a multi-turn dialogue, truncate the sample to the question containing the first instance of the identity term"""
    # get the first occurence of the identity term
    # find the first instance of 'Assistant' that occurs after the identity term
    id_term_idx = prompt.lower().index(term)
    assistant_idx = prompt[id_term_idx:].index('\n\nAssistant:')
    short_prompt = prompt[:id_term_idx + assistant_idx + len('\n\nAssistant:')]

    return short_prompt



def get_hh_samples_with_id_terms(data_dir: List[str], split: str, cache_dir: str = None) -> Dict[str, List[str]]:
    """Load the prompts from the Anthropic Helpful-Harmless dataset.

       Prompts are be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    assert len(data_dirs) > 0, 'Must provide at least one data directory'

    print(f'Loading HH ({split} split) from {data_dir}...')
    dataset = load_dataset('Anthropic/hh-rlhf', split=split, data_dir=data_dir, cache_dir=cache_dir)
    print('Done.')

    # Preprocess the dataset to prepare for TRL training
    original_columns = dataset.column_names
    dataset = dataset.map(split_prompt_and_responses, batched=True,remove_columns=original_columns)

    # filter identity terms from (Dixon, 2018)
    id_terms = ["atheist", "queer", "gay", "transgender", "lesbian", "homosexual", "feminist", "black people", "white people",
                "heterosexual", "islam", "muslim", "bisexual"]

    # Short prompt ends at the question with the first occurence of the identity term
    samples_with_id_term = {'prompt': [], 'short_prompt': [], 'chosen': [], 'rejected': []}

    for sample in tqdm(dataset):
        for term in id_terms:
            if term in sample['prompt'].lower():
                    short_prompt = truncate_prompt(sample['prompt'], term)
                    if short_prompt not in samples_with_id_term['short_prompt']:
                        samples_with_id_term['prompt'].append(sample['prompt'])
                        samples_with_id_term['short_prompt'].append(short_prompt)
                        samples_with_id_term['chosen'].append(sample['chosen'])
                        samples_with_id_term['rejected'].append(sample['rejected'])

    return samples_with_id_term

if __name__ == '__main__':
    data_dirs = ['helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    splits = ['train', 'test']
    hh_cache_dir = None
    samples_dir = './data/hh_helpful_with_id_terms'

    for data_dir in data_dirs:
        for split in splits:
            samples = get_hh_samples_with_id_terms(data_dir, split=split, cache_dir=hh_cache_dir)

            pd.DataFrame(samples).to_csv(f'{samples_dir}/hh-{data_dir}_{split}_with_id_terms.csv')
