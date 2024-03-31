import torch
import transformers
import sys


def load_tokenizer_and_model(name_or_path, model_checkpoint=None, return_tokenizer=False):
    """Load a model and (optionally) a tokenizer for inference"""
    assert name_or_path in ['gpt2-large', 'EleutherAI/pythia-2.8b'], "name_or_path must be in ['gpt2-large', 'EleutherAI/pythia-2.8b']"

    model = transformers.AutoModelForCausalLM.from_pretrained(name_or_path)

    if model_checkpoint is not None:
        print(f'Loading model from {model_checkpoint}...')
        model.load_state_dict(torch.load(model_checkpoint)['state'])
        print('Done.')

    else:
        print(f'No model checkpoint specified. Loading default {name_or_path} model.')

    if return_tokenizer:
        print('Loading tokenizer...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    return model



if __name__=='__main__':

    test_gpt2 = True
    test_pythia28 = True

    # Test the GPT2-large model trained with LHF
    if test_gpt2:
        print('='*80)
        # Test the gpt2-large model
        model_name = 'gpt2-large'
        model_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/gpt2l_dpo_gh_readme_params.pt'
        tokenizer, model = load_tokenizer_and_model(model_name, model_checkpoint=model_checkpoint)

        # Test the model on a prompt
        print(f'Testing {model_name}...')

        prompt = 'Who is Messi?'

        # Generate question based on the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, max_length=128, num_return_sequences=1, temperature=0.7)

        # Decode the generated question
        generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

        print('Response:', generated_question)

    # Test the Pythia 2.8B model trained with LHF
    if test_pythia28:
        print('='*80)
        # Test the Pythia 2.8B model
        model_name = 'EleutherAI/pythia-2.8b'
        model_checkpoint = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/test/pythia28_dpo_gh_readme_params.pt'
        tokenizer, model = load_tokenizer_and_model(model_name, model_checkpoint=model_checkpoint)

        # Test the model on a prompt
        print(f'Testing {model_name}...')

        prompt = 'Who is Messi?'

        # Generate question based on the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, max_length=128, num_return_sequences=1, temperature=0.7)

        # Decode the generated question
        generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

        print('Response:', generated_question)


