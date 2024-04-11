"""
    Author: @j-c-carr
    Script to evaluate models on the RealToxicityPrompts dataset
"""
import torch
from typing import List
from torch.utils.data import DataLoader
from transformers import pipeline
from inference_datasets import load_dataset
from inference import load_tokenizer_and_model

import pandas as pd


def get_toxicity_classifier(classifier_name="tomh/toxigen_roberta"):
    """Load either HateBERT_ToxiGen or RoBERTa_ToxiGen toxicity classifier from https://github.com/microsoft/TOXIGEN"""

    assert classifier_name in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"], \
        f'{classifier_name} must be in ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"]'

    classifier = pipeline("text-classification", model=classifier_name, device_map='auto')

    return classifier

def classify_outputs(dataloader, classifier_name="tomh/toxigen_roberta"):