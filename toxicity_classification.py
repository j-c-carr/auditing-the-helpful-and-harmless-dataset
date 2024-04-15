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
        toxicity_clf = pipeline("text-classification", model=classifier_name, device=0)
    else:
        toxicity_clf = pipeline("text-classification", model=classifier_name)

    print('Classifying toxicity of outputs...')
    toxicity_probs = []
    for batch in tqdm(dataloader):
        batch_scores = toxicity_clf(batch)
        toxicity_probs.extend([d['score'] if d['label'] == 'LABEL_1' else 1 - d['score'] for d in batch_scores])

    print('Done.')

    return toxicity_probs

if __name__ == '__main__':

    output_df = pd.read_csv('out/fairprism_eval/fairprism_eval_v1.csv', encoding='ascii', encoding_errors='replace')
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(output_df.columns)
    print(output_df.dtypes)
    output_df = output_df.astype(str)
    print(output_df.dtypes)
    print(output_df['prompts'].astype(str).tolist())
    exit(0)
    output_df = output_df.drop(columns=[col for col in output_df.columns if col.startswith('Unnamed')])
    print(output_df.columns)
    for col in output_df.columns:
        if col == 'prompts':
            continue
        col = 'ft_generations'
        print('='*80)
        print(f'{col} toxicity\n')
        dataloader = DataLoader(output_df[col].tolist()[:8], batch_size=batch_size, shuffle=False)  # Do NOT shuffle!
        toxicity_probs = classify_outputs(dataloader, device=device)
        print(toxicity_probs)

        output_df[f'{col}_toxicity_probs'] = toxicity_probs

    #output_df.to_csv('out/fairprism_eval/fairprism_eval_v1_with_tox.csv', index=False)
