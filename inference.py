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
        self.parser.add_argument('--base_model_name', type=str, default="", help='Base model name')
        self.parser.add_argument('--model_checkpoint', type=str, default=None,
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--model_name', type=str, default="base", help='Name of the model')
        # Prompt dataset args
        self.parser.add_argument('--dset_name', type=str, default='rtp',
                                 help='Model checkpoint. If None, generates with base model')
        self.parser.add_argument('--dset_split', type=str, default='train', choices=['train'],
                                 help="Prompt dataset split")
        self.parser.add_argument('--num_samples', type=int, default=None,
                                 help="Number of prompts to evaluate. If None, use the full prompt dataset")
        self.parser.add_argument('--dset_dir', type=str, default=None,
                                 help="Directory of the dataset. For HuggingFace datasets, if dset_dir is None then it"
                                      "defaults to the $HF_DATASETS_CACHE")
        # Model inference args
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generations')
        self.parser.add_argument('--num_return_sequences', type=int, default=1,
                                 help='Number of generations per prompt')
        self.parser.add_argument('--top_p', type=float, default=0.90, help='Param for top p filtering')
        self.parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for inference')
        self.parser.add_argument('--max_new_tokens', type=int, default=20,
                                 help='Max number of tokens per return sequence')

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


def main(args):
    """Main loop for performing inference"""

    # Load the prompts
    _prompts, focus = get_prompts(args.dset_name, split=args.dset_split, num_samples=args.num_samples,
                                 cache_dir=args.dset_dir)
    _ids = [i for i in range(len(_prompts))]
    prompts = [prompt for prompt in _prompts for _ in range(args.num_return_sequences)]
    prompt_ids = [_id for _id in _ids for _ in range(args.num_return_sequences)]

    # Add instruction format to the prompts
    formatted_prompts = add_instruction_format(prompts, dset_name=args.dset_name)

    # Generate outputs
    outputs = {}
    if "orpo_gpt2" in args.model_checkpoint:
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained()
        model = vllm.LLM(model=args.model_checkpoint, tokenizer="openai-community/gpt2-large")
    if "orpo_pythia2" in args.model_checkpoint:
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained()
        model = vllm.LLM(model=args.model_checkpoint, tokenizer="EleutherAI/pythia-2.8b")
    else:
        model = vllm.LLM(model=args.model_checkpoint)
    model_generations = model.generate(formatted_prompts,
        vllm.SamplingParams(top_p=args.top_p,
                            temperature=args.temperature,
                            max_tokens=args.max_new_tokens))
    model_generations_edited = [
        o.outputs[0].text for o in model_generations
    ]
    print('original model generations size:', len(model_generations))
    print("...generations done")
    print('total num of model generations:', len(model_generations_edited))

    outputs[f'{args.model_name}_generations'] = model_generations_edited


    if args.classify_toxicity:
        # Remove inference model from memory to save space
        del model
        model = None
        torch.cuda.empty_cache()
        gc.collect()

        toxicity_scores = classify_outputs(model_generations_edited,
                                           batch_size=args.batch_size,
                                           device=args.device)
        outputs[f'{args.model_name}_toxicity'] = toxicity_scores

    outputs['prompts'] = prompts
    outputs['prompt_ids'] = prompt_ids 

    # Save the focus for XSTest (for RTP, save the toxicity of the prompt)
    if len(focus) > 0:
        if args.dset_name in ["rtp", "xs-id-terms"]:
            # Focus is a dict
            for k, v in focus.items():
                outputs[f'prompt_{k}'] = [val for val in v for _ in range(args.num_return_sequences)]
        else:
            focus = [f for f in focus for _ in range(args.num_return_sequences)]
            outputs['focus'] = focus

    import os
    save_dir = f'out/{args.dset_name}_eval'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{args.base_model_name.replace("/", "-")}_{args.model_name}_{args.dset_name}_{args.num_return_sequences}_sequences.csv')
    print(f"Saving results to {path}...")
    pd.DataFrame(outputs).to_csv(path)

    if model is not None:
        # Remove inference model
        del model
        model = None
        torch.cuda.empty_cache()


if __name__ == '__main__':

    args = Args()
    args.parse()

    gc.collect()
    torch.cuda.empty_cache()

    # HF_TOKEN environment variable must be set to a hugging face access token
    login(token=os.environ["HF_TOKEN"])
    main(args)
