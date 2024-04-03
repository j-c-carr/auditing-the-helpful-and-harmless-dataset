"""
    Author: @j-c-carr
    Script to evaluate models on the RealToxicityPrompts dataset
"""
import torch
from typing import List
from torch.utils.data import DataLoader
from transformers import pipeline
from datasets import load_dataset
from inference import load_tokenizer_and_model

import pandas as pd


def get_toxicity_classifier(classifier_name="tomh/toxigen_roberta"):
    """Load either HateBERT_ToxiGen or RoBERTa_ToxiGen toxicity classifier from https://github.com/microsoft/TOXIGEN"""

    assert classifier_name in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"], \
        f'{classifier_name} must be in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"]'

    classifier = pipeline("text-classification", model=classifier_name, device_map='auto')

    return classifier


def get_rtp_dataset(cache_dir=None, device='cpu'):
    """Fetches the RealToxicityPrompts dataset from HuggingFace"""
    print(f'Loading RealToxicityPrompts dataset from Huggingface...')
    dataset = load_dataset('allenai/real-toxicity-prompts', split='train', cache_dir=cache_dir)
    dataset.set_format(type="torch", device=device)
    print('Done.')

    return dataset


def get_rtp_dataloader(batch_size=32, cache_dir=None, device='cpu'):
    """Creates a DataLoader for the RealToxicityPrompts dataset."""
    rtp_dataset = get_rtp_dataset(cache_dir=cache_dir, device=device)

    def collate_fn(batch):
        """Custom collate function which loads only the prompt text and toxicity score for each sample"""
        return {'prompt': [b['prompt']['text'] for b in batch]}

    dataloader = DataLoader(rtp_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader


def add_instruction_prefix(prompts: List[str]) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    return [f'\n\nHuman: Continue the following prompt: {prompt}\n\nAssistant:' for prompt in prompts]


def rtp_eval(base_pipeline, ft_pipeline, dataloader, num_batches=1, classifier_name="tomh/toxigen_roberta"):
    """Evaluates the toxicity of a model's generations on the RealToxicityPrompts dataset."""

    prompts = []
    prompt_toxicity_scores = []     # toxicity of prompt, calculated using PerspectiveAPI
    base_generations = []
    ft_generations = []
    base_toxicity_scores = []       # toxicity of base model output, calculated with toxicity_clf
    ft_toxicity_scores = []         # toxicity of fine-tuned model output, calculated with toxicity_clf

    # Get the toxicity classifier
    toxicity_clf = get_toxicity_classifier(classifier_name)

    for i, batch in enumerate(dataloader):
        
        # Collect :batch_size: x :num_batches: examples
        if i == num_batches:
            break

        # Generate from base model and fine-tuned model
        base_prompts = batch['prompt']
        ft_prompts = add_instruction_prefix(batch['prompt'])


        # For simplicity, use the same kwargs for both the base and fine-tuned model pipelines
        generator_kwargs = {'pad_token_id': base_pipeline.tokenizer.eos_token_id,
                            'max_new_tokens': 32}

        # Calculate toxicity of base model outputs
        _base_generations = base_pipeline(base_prompts, **generator_kwargs)
        _base_generations = [_base_generations[i][0]['generated_text'][len(base_prompts[i]):] for i in range(len(base_prompts))]
        _base_toxicity_scores = toxicity_clf(_base_generations)

        # Calculate toxicity of fine-tuned model outputs
        _ft_generations = ft_pipeline(ft_prompts, **generator_kwargs)
        _ft_generations = [_ft_generations[i][0]['generated_text'][len(ft_prompts[i]):] for i in range(len(ft_prompts))]
        _ft_toxicity_scores = toxicity_clf(_ft_generations)

        # Record the prompts, generations and toxicity scores
        prompts += batch['prompt']
        _prompt_toxicity_scores = toxicity_clf(batch['prompt'])
        prompt_toxicity_scores += [d['score'] if d['label'] == 'LABEL_1' else 1 - d['score'] for d in _prompt_toxicity_scores]

        base_generations += _base_generations
        ft_generations += _ft_generations

        base_toxicity_scores += [d['score'] if d['label'] == 'LABEL_1' else 1 - d['score'] for d in _base_toxicity_scores]
        ft_toxicity_scores += [d['score'] if d['label'] == 'LABEL_1' else 1 - d['score'] for d in _ft_toxicity_scores]

    return {'prompts': prompts,
            'base_generations': base_generations,
            'ft_generations': ft_generations,
            'base_toxicity_scores': base_toxicity_scores,
            'ft_toxicity_scores': ft_toxicity_scores,
            'prompt_toxicity_scores': prompt_toxicity_scores}


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Total number of sample generations is :batch_size: * :num_batches:
    batch_size = 64
    num_batches = 4

    # Load RealToxicityPrompts dataset and create dataloader
    cache_dir = '/network/scratch/j/jonathan.colaco-carr'
    dataloader = get_rtp_dataloader(batch_size=batch_size, cache_dir=cache_dir, device=device)

    # Load models
    gpt2l_dpo_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/gpt2l_dpo_gh_readme_params.pt'
    pythia28_dpo_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/pythia28_dpo_gh_readme_params.pt'

    # Tuple of (model_name, model_checkpoint)
    models = [('gpt2-large', gpt2l_dpo_checkpoint)]
#              ('EleutherAI/pythia-2.8b', pythia28_dpo_checkpoint)]

    for model_name, model_checkpoint in models:
        print('='*80)
        print(f'Evaluating {model_name}...')

        # Load the base model
        base_pipeline = pipeline("text-generation", model=model_name, device=0)

        # Load the fine-tuned model with same the tokenizer as the base model
        ft_model = load_tokenizer_and_model(model_name, model_checkpoint=model_checkpoint, device=device)
        ft_pipeline = pipeline("text-generation", model=ft_model, tokenizer=model_name, device=1)

        # Evaluate the base model and fine-tuned model on RealToxicityPrompts
        eval_output = rtp_eval(base_pipeline, ft_pipeline, dataloader, num_batches=num_batches)
        eval_output['model_name'] = [model_name for _ in range(len(eval_output['prompts']))]

        # Save results
        eval_output_file = f'/network/scratch/j/jonathan.colaco-carr/hh_fruits/out/samples/{model_name.replace("/","-")}_rtp_samples.csv'
        print(f'Saving evaluation results to {eval_output_file}')
        eval_df = pd.DataFrame(eval_output)
        eval_df.to_csv(eval_output_file)
        print('Done.')
