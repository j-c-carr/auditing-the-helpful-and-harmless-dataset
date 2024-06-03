from tqdm import tqdm
import torch
import transformers
import pandas as pd
from torch.utils.data import DataLoader

from utils import (create_id_term_prompts, get_baseline_prompts, create_id_term_v2fact_prompts,
get_factv2_baseline_prompts, get_factv3_baseline_prompts, create_id_term_v3fact_prompts,)
from inference_datasets import get_prompts, add_instruction_format, get_prompts_from_csv
from toxicity_classification import classify_outputs


def load_tokenizer_and_model(name_or_path, model_checkpoint=None, return_tokenizer=False, device='cpu'):
    """Load a model and (optionally) a tokenizer for inference"""
    assert name_or_path in ['gpt2-large', 'EleutherAI/pythia-2.8b'], "name_or_path must be in ['gpt2-large', 'EleutherAI/pythia-2.8b']"

    model = transformers.AutoModelForCausalLM.from_pretrained(name_or_path)

    if model_checkpoint is not None:
        print(f'Loading model checkpoint from {model_checkpoint}...')
        model.load_state_dict(torch.load(model_checkpoint)['state'])
        print('Done.')

    else:
        print(f'No model checkpoint specified. Loading default {name_or_path} model.')

    device = torch.device(device)
    model.to(device)

    if return_tokenizer:
        print('Loading tokenizer...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    return model

@torch.no_grad()
def inference_loop(model, tokenizer, prompt_dataloader, device='cpu', instruction_format=False, **generate_kwargs):
    """Use model and tokenizer to tokenizer"""
    model.eval()

    prompts = []
    outputs = []

    print('Generating outputs...')
    for batch_prompts in tqdm(prompt_dataloader):
        prompts.extend(batch_prompts)

        # Add "Human: ... Assistant: ..." for models fine-tuned on Helpful-Harmless dataset
        if instruction_format:
            batch_prompts = add_instruction_format(batch_prompts, dset_name='xstest')

        inputs = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors='pt').to(device)
        batch_outputs = model.generate(**inputs, **generate_kwargs)

        outputs.extend([batch_outputs[i, inputs['input_ids'].shape[1]:] for i in range(len(batch_prompts))])
    print('Done.')

    # Todo: decode outputs
    print('Decoding outputs...')
    for i in range(len(outputs)):
        outputs[i] = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print('Done.')

    return prompts, outputs 

if __name__=='__main__':

    # :cache_dir: is the folder containing the dataset. For rtp and hh, set this equal to the hugging face cache
    # folder, e.g.'/network/scratch/j/jonathan.colaco-carr'
    # For FairPrism, or XSTest set :cache_dir: equal to the folder containing the dataset csv file
    dset_name = 'xstest'
    split = 'train'
    #cache_dir = '/network/scratch/j/jonathan.colaco-carr/.cache/jonathan.colaco-carr'
    cache_dir = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/data/xstest'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = None # if None, use the full dataset
    batch_size = 64
    max_new_tokens = 80

    base_model_name = 'EleutherAI/pythia-2.8b'

    classify_toxicity = False
    tox_clf_batch_size = 16     # use a different batch size for toxicity classifier

    ###############################################
    # Add models for inference here:
    checkpoint_dir = "/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/v1_checkpoints/pythia28"

    # keys are the model name, values are the path to the model's weights
    #models = {#'base_lm': None,
    #          'help_only_dpo_longer': f'{checkpoint_dir}/helpful_only/dpo_gpt2l_helpful_longer_2024-04-13_step-200000_policy.pt',
    #          #'hh_filtered_dpo_longer': f'{checkpoint_dir}/all_filtered/gpt2l_dpo_filtered_longer_2024-04-14_step-280000_policy.pt',
    #          'hh_full_dpo_longer': f'{checkpoint_dir}/hh_full/dpo_gpt2l_paper_params_longer_2024-04-13_step-240000_policy.pt'}
    models = {'base_lm': None,
              'help_only': f'{checkpoint_dir}/helpful_only/dpo_pythia28_helpful_only_2024_04_16_step-160000.pt',
              'hh_filtered': f'{checkpoint_dir}/all_filtered/dpo_pythia28_filtered_2024-04-16_step-160000_policy.pt',
              'hh_full': f'{checkpoint_dir}/hh_full/dpo_pythia28_hh_full_1_epoch.pt'}
    ###############################################

    # Load the prompts
    prompts = get_prompts(dset_name, split=split, num_samples=num_samples, cache_dir=cache_dir)
    #prompts = create_id_term_v3fact_prompts() #create_id_term_prompts() #get_prompts_from_csv(dset_path, 'short_prompt')
    prompt_dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)  # DO NOT SHUFFLE!

    # Load the base model
    tokenizer, model = load_tokenizer_and_model(base_model_name, return_tokenizer=True, device=device)

    outputs = {}
    for model_name, model_checkpoint in models.items():
        print('='*80)
        print('Model: ', model_name)

        # Load the model weights
        if model_checkpoint is not None:
            print(f'Loading model checkpoint from {model_checkpoint}...')
            model.load_state_dict(torch.load(model_checkpoint)['state'])
            print('Done.')

        # Add instruction format for fine-tuned models
        # HH dataset already has the proper instruction format
        if (model_checkpoint is not None) and ('hh' not in dset_name):
            instruction_format = True
        else:
            instruction_format = False

        # Generate model outputs
        prompts, model_generations = inference_loop(model, tokenizer,
                                                    prompt_dataloader,
                                                    instruction_format=instruction_format,
                                                    device=device,
                                                    max_new_tokens=max_new_tokens)

        outputs[f'{model_name}_generations'] = model_generations

        # Classify the toxicity of the model generations
        if classify_toxicity:
            toxicity_probs = classify_outputs(model_generations, tox_clf_batch_size, device=device)
            outputs[f'{model_name}_generations_toxicity_probs'] = toxicity_probs

        pd.DataFrame(outputs).to_csv(f'{base_model_name.replace("/","-")}_{dset_name}_prompts_{max_new_tokens}_tokens.csv')

    # Save the prompts and outputs
    outputs['prompts'] = prompts
    pd.DataFrame(outputs).to_csv(f'{base_model_name.replace("/","-")}_{dset_name}_prompts_{max_new_tokens}_tokens.csv')
