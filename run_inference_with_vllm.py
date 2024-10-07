"""
Main script for generating model outputs on a dataset of test prompts
"""
import argparse
import os
from tqdm import tqdm
import torch
import transformers
import pandas as pd
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM
from inference_datasets import get_prompts, add_instruction_format
from huggingface_hub import login

from toxicity_classification import classify_outputs

import vllm

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
                                 choices=["EleutherAI/pythia-2.8b", "gpt2-large", "meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.1"],
                                 help='Base model name')
        self.parser.add_argument('--model_checkpoint', type=str, default=None,
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--model_name', type=str, default="base", help='Name of the model')
        # Prompt dataset args
        self.parser.add_argument('--dset_name', type=str, default='xstest-plus',
                                 choices=['xstest', 'xstest-plus', 'rtp'],
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--dset_split', type=str, default='train', choices=['train'],
                                 help="Prompt dataset split")
        self.parser.add_argument('--num_samples', type=int, default=None,
                                 help="Number of prompts to evaluate. If None, use the full prompt dataset")
        self.parser.add_argument('--dset_dir', type=str, default=None,
                                 help="Directory of the dataset. For HuggingFace datasets, if dset_dir is None then it"
                                      "defaults to the $HF_DATASETS_CACHE")
        # Model inference args
        self.parser.add_argument('--seed', type=int, default=1, help='Random seed for model generations')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generations')
        self.parser.add_argument('--num_return_sequences', type=int, default=1,
                                 help='Number of generations per prompt')
        self.parser.add_argument('--top_p', type=float, default=0.95, help='Param for top p filtering')
        self.parser.add_argument('--top_k', type=int, default=50, help='Param for top k filtering')
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

        # Set each of the parser's attributes as class attributes
        for key, value in vars(args).items():
            setattr(self, key, value)

        # Kwargs passed to model.generate()
        self.generator_kwargs = {"max_new_tokens": args.max_new_tokens,
                                 "do_sample": args.do_sample,
                                 "num_return_sequences": args.num_return_sequences,
                                 "top_k": args.top_k,
                                 "top_p": args.top_p}


def load_tokenizer_and_model(base_model_name, model_checkpoint=None, return_tokenizer=False, device='cpu'):
    """Load a model and (optionally) a tokenizer for inference"""

    # Load big models with Peft
    if base_model_name in ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.1"]:
        model = AutoPeftModelForCausalLM.from_pretrained(model_checkpoint,  # directory of saved model
                                                         low_cpu_mem_usage=True,
                                                         torch_dtype=torch.float16,
                                                         # load_in_4bit=True,
                                                         is_trainable=False)

    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name)

        if model_checkpoint is not None:
            print(f'Loading model checkpoint from {model_checkpoint}...')
            model.load_state_dict(torch.load(model_checkpoint)['state'])
            print('Done.')

        else:
            print(f'No model checkpoint specified. Loading base {base_model_name} model.')

    model.to(torch.device(device))

    if return_tokenizer:
        print('Loading tokenizer...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,
                                                               padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    return model


@torch.no_grad()
def inference_loop(model, tokenizer, prompt_dataloader, device='cpu', instruction_format=False, **generate_kwargs):
    """Use model and tokenizer to tokenizer"""
    model.eval()

    outputs = []

    print('Generating outputs...')
    for batch_prompts in tqdm(prompt_dataloader):

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
    print("Len outputs: ", len(outputs))

    return outputs 


def main(args):
    """Main loop for performing inference"""

    # Load the prompts
    prompts, focus = get_prompts(args.dset_name, split=args.dset_split, num_samples=args.num_samples,
                                 cache_dir=args.dset_dir)
    # DO NOT SHUFFLE!
    prompt_dataloader = DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

    # Load the tokenizer and model
    #tokenizer, model = load_tokenizer_and_model(args.base_model_name, model_checkpoint=args.model_checkpoint,
    #                                            return_tokenizer=True, device=args.device)
    model = vllm.LLM(model=args.model_checkpoint)

    # Add instruction format to the prompts if necessary. Since the prompts in the hh dataset are already formated as
    # questions, we don't change them
    if 'hh' in args.dset_name:
        instruction_format = False
    else:
        instruction_format = True 

    print("TODO: ADD INSTRUCTION FORMAT")
    outputs = {}
    model_generations = model.generate(
        [
            prompt for prompt in prompts
            for _ in range(args.num_return_sequences)
        ],
        vllm.SamplingParams(top_p=args.top_p,
                            top_k=args.top_k,
                            max_tokens=args.max_new_tokens,
                            seed=args.seed))
    model_generations_edited = [
        o.outputs[0].text for o in model_generations
    ]
    print('original model generations size:', len(model_generations))
    print("...generations done")
    print('total num of model generations:', len(model_generations_edited))

    outputs[f'{args.model_name}_generations'] = model_generations_edited


    if args.classify_toxicity:
        toxicity_scores = classify_outputs(model_generations,
                                           batch_size=args.batch_size)
        outputs[f'{args.model_name}_toxicity'] = toxicity_scores

    prompts = [prompt for prompt in prompts for _ in range(args.num_return_sequences)]
    outputs['prompts'] = prompts

    # Save the focus for XSTest
    if len(focus) > 0:
        focus = [f for f in focus for _ in range(args.num_return_sequences)]
        outputs['focus'] = focus

    import os
    save_dir = f'out/{args.dset_name}_eval'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{args.base_model_name.replace("/", "-")}_{args.model_name}_{args.dset_name}_{args.num_return_sequences}_sequences.csv')
    print(f"Saving results to {path}...")
    pd.DataFrame(outputs).to_csv(path)


if __name__ == '__main__':

    args = Args()
    args.parse()

    # HF_TOKEN environment variable must be set to a hugging face access token
    login(token=os.environ["HF_TOKEN"])
    main(args)
