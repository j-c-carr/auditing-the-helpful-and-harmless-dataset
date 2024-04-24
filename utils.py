"""
Utility script containing various helper functions
Author: @j-c-carr
"""
from typing import List, Dict, Tuple
from datasets import DatasetDict
from transformers import AutoTokenizer


def extract_hh_prompt_from_sample(prompt_and_response: str) -> str:
    """Extract the prompt from a prompt-response pair from the HH dataset."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def split_prompt_and_responses(samples) -> Dict[str, List[str]]:
    """Preprocesses data"""
    prompt = [extract_hh_prompt_from_sample(chosen_str) for chosen_str in samples['chosen']]
    chosen_response = [chosen_str[len(prompt[i]):] for i, chosen_str in enumerate(samples['chosen'])]
    rejected_response = [rejected_str[len(prompt[i]):] for i, rejected_str in enumerate(samples['rejected'])]
    return {'prompt': prompt, 'chosen': chosen_response, 'rejected': rejected_response}


def get_dataset_statistics(dataset: DatasetDict, tokenizer: AutoTokenizer) -> Tuple[float, float, float]:
    """Compute the average number of tokens in the prompt, chosen respons and rejected response"""
    prompt_tokens, = 0
    chosen_tokens = 0
    rejected_tokens = 0

    for example in dataset:

        tokens = tokenizer(example['prompt'], return_length=True)
        prompt_tokens += tokens['length']

        tokens = tokenizer(example['chosen'], return_length=True)
        chosen_tokens += tokens['length']

        tokens = tokenizer(example['rejected'], return_length=True)
        rejected_tokens += tokens['length']

    return prompt_tokens / len(dataset), chosen_tokens / len(dataset), rejected_tokens / len(dataset)

