from tqdm import tqdm
import torch
import transformers

from torch.utils.data import DataLoader

from inference_datasets import get_prompts


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
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path).to(device)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    return model

@torch.no_grad()
def inference(model, tokenizer, prompt_dataloader, device='cpu', **generate_kwargs):
    """Use model and tokenizer to tokenizer"""

    prompts = []
    outputs = []    # tokenized outputs

    print('Generating outputs...')
    for batch_prompts in tqdm(prompt_dataloader):
        inputs = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors='pt').to(device)
        batch_outputs = model.generate(inputs['input_ids'], **generate_kwargs).to(device)
        outputs.extend(batch_outputs[:, inputs['input_ids'].shape[1]:])
    print('Done.')

    # Todo: decode outputs
    generations = []
    print('Decoding outputs...')
    for output in outputs:
        generations.extend(tokenizer.decode(output))
    print('Done.')
    return prompts, outputs

if __name__=='__main__':

    # Load the BERT tokenizer and language model
    from transformers import BertForMaskedLM, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    cache_dir = None    # todo: change for cluster
    prompts = get_prompts("rtp", num_samples=1000, instruction_format=False, cache_dir=cache_dir)
    prompt_dataloader = DataLoader(prompts, batch_size=2, shuffle=False)

    inference(model, tokenizer, prompt_dataloader, max_new_tokens=35)

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


