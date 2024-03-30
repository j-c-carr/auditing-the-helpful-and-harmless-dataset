"""
    Author: @j-c-carr
    Evaluates model on real toxicity prompts using toxicity classifiers trained on ToxiGen
"""
import torch
from torch.utils.data import DataLoader
from transformers import pipeline
from datasets import load_dataset


def get_toxicity_classifier(classifier_name="tomh/toxigen_roberta"):
    """Loads either hatebert or roberta models which have been fine-tuned on ToxiGen.
    Loading models following: https://github.com/microsoft/TOXIGEN"""

    assert classifier_name in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"], f'{classifier_name} must be in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"]'

    classifier = pipeline("text-classification", model=classifier_name)

    return classifier

def get_rtp_dataset(cache_dir=None):
    """Fetches the RealToxicityPrompts dataset from HuggingFace"""
    print(f'Loading RealToxicityPrompts dataset from Huggingface...')
    dataset = load_dataset('allenai/real-toxicity-prompts', split='train', cache_dir=cache_dir)
    print('done')
    return dataset

def get_rtp_dataloader(batch_size=32):
    """Creates a DataLoader from the RealToxicityPrompts dataset."""
    rtp_dataset = get_rtp_dataset()

    # Do NOT shuffle the training dataset
    dataloader = DataLoader(rtp_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def rtp_eval(model, tokenizer=None, classifier_name="tomh/toxigen_roberta"):
    """Evaluates the toxicity of a model's generations on the RealToxicityPrompts dataset."""
    classifier = get_toxicity_classifier(classifier_name)
    dataloader = get_rtp_dataloader()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    toxicity_scores = []
    for batch in dataloader:
        batch_prompts = batch['prompt']['text']
        generated_text = generator(batch_prompts)
        generated_text = [gt[0]['generated_text'] for gt in generated_text]
        _toxicity_scores = classifier(generated_text)
        toxicity_scores += _toxicity_scores

if __name__ == '__main__':
    toxigen = load_dataset("skg/toxigen-data", split='train')

    # Todo: try out with GPT-2 and Pythia models.
    # Todo: try out with GPT-2 SFT and Pythia SFT models
    # Todo: try out with GPT-2 DPO and Pythia DPO models
    rtp_eval('bert-base-uncased')



