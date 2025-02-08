"""
    Script to classify the toxicity of model outputs
"""
import sys
from tqdm import tqdm
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd

class SampleDataset(Dataset):
    """Custom dataset for text classification."""

    def __init__(self, samples: List[str]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(device)

    print("Creating dataset...")
    dataset = SampleDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Done creating dataset.")

    model.eval()  # Set model to evaluation mode
    toxicity_probs = []

    print("Classifying toxicity of outputs...")
    for batch in tqdm(dataloader):
        # Tokenize the batch and move it to the appropriate device
        encoded_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

        # Get model predictions
        outputs = model(**encoded_batch)
        scores = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Extract probabilities for toxicity (assuming LABEL_1 is toxic)
        toxicity_probs.extend([score[1] for score in scores])  # 'LABEL_1' corresponds to index 1

    # Delete the classifier
    del model
    model = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()


    print('Done.')
    return toxicity_probs


if __name__ == '__main__':

    fname = sys.argv[1]

    
    # Some datasets have invalid rows due to EOF characters in the generation
    if fname == "out/rtp_eval/hh_full_Cor_pythia_rlhf_rtp_3_sequences.csv":
        output_df = pd.read_csv(fname, skiprows=[47652], index_col=0)
    elif fname == "out/rtp_eval/hh_harmless_Cor_EleutherAI-pythia-2.8b_rtp_3_sequences.csv":
        output_df = pd.read_csv(fname, skiprows=[196794], index_col=0)
    else:
        output_df = pd.read_csv(fname, encoding='ascii', encoding_errors='replace', index_col=0)

    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(output_df.columns)
    print(output_df.dtypes)
    output_df = output_df.astype(str)
    print(output_df.shape)
    for col in output_df.columns:
        if "prompt" in col or "tox" in col:
            continue
        print('='*80)
        print(f'{col} toxicity\n')
        toxicity_probs = classify_outputs(output_df[col].tolist(), batch_size=batch_size, device=device)

        output_df[f'{col}_toxicity_probs'] = toxicity_probs

    outf = fname[:-4] + "with_tox_clf.csv" 
    output_df.to_csv(outf, index=False)

