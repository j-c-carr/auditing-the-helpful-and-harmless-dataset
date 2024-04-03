from tqdm import tqdm
import torch
import transformers
import pandas as pd

from torch.utils.data import DataLoader

from inference_datasets import get_prompts, add_instruction_format


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
            batch_prompts = add_instruction_format(batch_prompts)

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

def save_prompts_and_outputs(prompts, outputs, filename, base_outputs=None):
    """Saves prompts and outputs to csv file. If base model is also tested, include base outputs.
    NOTE: assumes that :outputs: and :base_outputs: are in the SAME order."""
    if base_outputs is not None:
        df = pd.DataFrame({'prompts': prompts, 'ft_outputs': outputs, 'base_outputs': outputs})
    else:
        df = pd.DataFrame({'prompts': prompts, 'outputs': outputs})

    assert filename[-4:] == '.csv', 'file must be a csv'
    df.to_csv(filename, index=False)

if __name__=='__main__':

    cache_dir = '/network/scratch/j/jonathan.colaco-carr' # todo: change for cluster
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    num_samples = 5_000
    batch_size = 128
    max_new_tokens=32

    # Load RealToxicityPrompts
    prompts = get_prompts("rtp", num_samples=num_samples, cache_dir=cache_dir)
    prompt_dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False) # DO NOT SHUFFLE!

    # Load the base model
    model_name = 'gpt2-large'
    tokenizer, model = load_tokenizer_and_model(model_name, model_checkpoint=None, return_tokenizer=True, device=device)

    # Inference with the base model
    prompts, base_outputs = inference_loop(model, tokenizer, prompt_dataloader, instruction_format=False, device=device, 
                                           max_new_tokens=max_new_tokens)

    # Load the fine-tuned model
    model_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/gpt2l_dpo_gh_readme_params.pt'
    print(f'Loading model checkpoint from {model_checkpoint}...')
    model.load_state_dict(torch.load(model_checkpoint)['state'])
    print('Done.')

    # Inference with the fine-tuned model
    prompts, outputs = inference_loop(model, tokenizer, prompt_dataloader, instruction_format=True, device=device,
                                      max_new_tokens=max_new_tokens)

    save_prompts_and_outputs(prompts, outputs, filename='test.csv', base_outputs=base_outputs)

    #test_gpt2 = True
    #test_pythia28 = True

    ## Test the GPT2-large model trained with LHF
    #if test_gpt2:
    #    print('='*80)
    #    # Test the gpt2-large model
    #    model_name = 'gpt2-large'
    #    model_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/gpt2l_dpo_gh_readme_params.pt'
    #    tokenizer, model = load_tokenizer_and_model(model_name, model_checkpoint=model_checkpoint)

    #    # Test the model on a prompt
    #    print(f'Testing {model_name}...')

    #    prompt = 'Who is Messi?'

    #    # Generate question based on the prompt
    #    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #    output = model.generate(input_ids, max_length=128, num_return_sequences=1, temperature=0.7)

    #    # Decode the generated question
    #    generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

    #    print('Response:', generated_question)

    ## Test the Pythia 2.8B model trained with LHF
    #if test_pythia28:
    #    print('='*80)
    #    # Test the Pythia 2.8B model
    #    model_name = 'EleutherAI/pythia-2.8b'
    #    model_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/pythia28_dpo_gh_readme_params.pt'
    #    tokenizer, model = load_tokenizer_and_model(model_name, model_checkpoint=model_checkpoint)

    #    # Test the model on a prompt
    #    print(f'Testing {model_name}...')

    #    prompt = 'Who is Messi?'

    #    # Generate question based on the prompt
    #    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #    output = model.generate(input_ids, max_length=128, num_return_sequences=1, temperature=0.7)

    #    # Decode the generated question
    #    generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

    #    print('Response:', generated_question)


