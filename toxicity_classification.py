"""
    Author: @j-c-carr
    Script to classify the toxicity of model outputs
"""
from tqdm import tqdm
import torch
from typing import List
from torch.utils.data import DataLoader
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd


def get_toxicity_classifier(classifier_name="tomh/toxigen_roberta", device='cpu'):
    """Load either HateBERT_ToxiGen or RoBERTa_ToxiGen toxicity classifier from https://github.com/microsoft/TOXIGEN"""

    assert classifier_name in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"], \
        f'{classifier_name} must be in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"]'

    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_name)

    device = torch.device(device)
    classifier.to(device)

    return tokenizer, classifier

@torch.no_grad()
def classify_outputs(samples: List[str], batch_size, device='cpu', classifier_name="tomh/toxigen_roberta"):
    """Classify model outputs using toxicity classifier"""

    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=False)  # Do NOT shuffle!

    if device != 'cpu':
        toxicity_clf = pipeline("text-classification", model=classifier_name, device=0, truncation=True)
    else:
        toxicity_clf = pipeline("text-classification", model=classifier_name, truncation=True)

    print('Classifying toxicity of outputs...')
    toxicity_probs = []
    for batch in tqdm(dataloader):
        batch_scores = toxicity_clf(batch)
        toxicity_probs.extend([d['score'] if d['label'] == 'LABEL_1' else 1 - d['score'] for d in batch_scores])

    print('Done.')

    return toxicity_probs

if __name__ == '__main__':

    f_name = 'EleutherAI-pythia-2.8b_fairprism_prompts_80_tokens'
    output_df = pd.read_csv(f'{f_name}.csv', index_col=0)
    print(output_df.columns)
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Number of prompts: ', output_df['prompts'].shape[0])
    # Get classify the toxicity of the prompts
    prompts = output_df['prompts'].tolist()
    toxicity_probs = classify_outputs(prompts, batch_size=16, device=device)

    output_df[f'prompts_toxicity_probs'] = toxicity_probs
    print(output_df.columns)

    output_df.to_csv(f'{f_name}_v2.csv', index=False)
