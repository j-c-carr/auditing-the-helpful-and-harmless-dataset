"""
Main script for generating model outputs on a dataset of test prompts
"""
import argparse
from tqdm import tqdm
import torch
import transformers
import pandas as pd
from torch.utils.data import DataLoader

from inference_datasets import get_prompts, add_instruction_format

from toxicity_classification import classify_outputs

import gc
gc.collect()
torch.cuda.empty_cache()


class Args:
    """Handles the command-line arguments"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Command Line Arguments")
        # Model name and checkpoint
        self.parser.add_argument('--base_model_name', type=str,
                                 default="EleutherAI/pythia-2.8b",
                                 choices=["EleutherAI/pythia-2.8b", 'gpt2-large'],
                                 help='Base model name')
        self.parser.add_argument('--model_checkpoint', type=str,
                                 default=None,
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--model_name', type=str,
                                 default="base", help='Name of the model')
        # Prompt dataset args
        self.parser.add_argument('--dset_name', type=str,
                                 default='xstest_plus',
                                 choices=['xstest', 'xstest_plus', 'rtp'],
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--dset_split', type=str,
                                 default='train', choices=['train'],
                                 help="Prompt dataset split")
        self.parser.add_argument('--num_samples', type=int,
                                 default=None,
                                 help="Number of prompts to evaluate. If None,"
                                      "use the full prompt dataset")
        self.parser.add_argument('--dset_dir', type=str,
                                 default=None,
                                 help="Directory of the dataset. For HuggingFace"
                                      "datasets, if dset_dir is None then it "
                                      "defaults to the $HF_DATASETS_CACHE")
        # Model inference args
        self.parser.add_argument('--seed', type=int, default=1,
                                 help='Random seed for model generations')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size for generations')
        self.parser.add_argument('--num_return_sequences', type=int, default=1,
                                 help='Number of generations per prompt')
        self.parser.add_argument('--top_p', type=float, default=0.95,
                                 help='Param for top p filtering')
        self.parser.add_argument('--top_k', type=int, default=50,
                                 help='Param for top k filtering')
        self.parser.add_argument('--max_new_tokens', type=int, default=50,
                                 help='Max number of tokens per return sequence')
        self.parser.add_argument('--do_sample', action='store_true',
                                 help='Sample non-deterministically from the model')

        # Toxicity classification
        self.parser.add_argument('--classify_toxicity', action='store_true',
                                 help='Classify toxicty of model outputs')

        # Set the torch device automatically
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def parse(self):
        """Parse command line arguments and assign them as attributes."""
        args = self.parser.parse_args()

        # Set each command line parameter as a class attribute
        for key, value in vars(args).items():
            setattr(self, key, value)

        # Kwargs passed to model.generate()
        self.generator_kwargs = {"max_new_tokens": args.max_new_tokens,
                                 "do_sample": args.do_sample,
                                 "num_return_sequences": args.num_return_sequences,
                                 "top_k": args.top_k,
                                 "top_p": args.top_p}

def load_tokenizer_and_model(name_or_path, model_checkpoint=None, return_tokenizer=False, device='cpu'):
    """Load a model and (optionally) a tokenizer for inference"""
    assert name_or_path in ['gpt2-large', 'EleutherAI/pythia-2.8b'], \
        "name_or_path must be in ['gpt2-large', 'EleutherAI/pythia-2.8b']"

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
        outputs.extend([batch_outputs[i, inputs['input_ids'].shape[1]:] for i in range(len(batch_outputs))])

    print('Done.')

    print('Decoding outputs...')
    for i in range(len(outputs)):
        outputs[i] = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print("Len prompts: ", prompts)
    print("Len outputs: ", len(outputs))

    return prompts, outputs 


def main(args):
    """Main loop for performing inference"""

    # Load the prompts
    if args.dset_name == 'xstest_plus':
        # Append custom prompts for xstest_plus
        prompts = get_prompts("xstest", split=args.dset_split,
                              num_samples=args.num_samples,
                              cache_dir=args.dset_dir)
        from xs_custom import disc_prompts, contrast_disc_prompts

        prompts.extend(disc_prompts)
        prompts.extend(contrast_disc_prompts)

    else:
        prompts = get_prompts(args.dset_name, split=args.dset_split,
                              num_samples=args.num_samples,
                              cache_dir=args.dset_dir)

    prompt_dataloader = DataLoader(prompts, batch_size=args.batch_size,
                                   shuffle=False)  # DO NOT SHUFFLE!

    # Load the base model
    tokenizer, model = load_tokenizer_and_model(args.base_model_name,
                                                return_tokenizer=True,
                                                device=args.device)

    # Load the model weights
    if args.model_checkpoint is not None:
        print(f'Loading model checkpoint from {args.model_checkpoint}...')
        state_dict = torch.load(args.model_checkpoint)['state']
        model.load_state_dict(state_dict)
        print('Done.')

    # Add instruction format to the prompts if necessary
    if (args.model_checkpoint is not None) and ('hh' not in args.dset_name):
        instruction_format = True
    else:
        instruction_format = False

    # Generate model outputs
    outputs = {}
    print(f"Generating answers to {len(prompts)} prompts...")
    prompts, model_generations = inference_loop(model, tokenizer,
                                                prompt_dataloader,
                                                instruction_format=instruction_format,
                                                device=args.device,
                                                **args.generator_kwargs)

    outputs[f'{args.model_name}_generations'] = model_generations

    if args.classify_toxicity:
        toxicity_scores = classify_outputs(model_generations,
                                           batch_size=args.batch_size)
        outputs[f'{args.model_name}_toxicity'] = toxicity_scores

    prompts = [prompt for prompt in prompts for _ in range(args.num_return_sequences)]

    # Save the prompts and outputs
    outputs['prompts'] = prompts
    pd.DataFrame(outputs).to_csv(
        f'out/{args.dset_name}_eval/'
        f'{args.base_model_name.replace("/", "-")}_{args.model_name}'
        f'_{args.dset_name}_{args.num_return_sequences}_sequences.csv')


if __name__ == '__main__':

    args = Args()
    args.parse()

    # Set seed for model generations
    transformers.set_seed(args.seed)

    main(args)
