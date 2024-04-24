# Auditing the Helpful and Harmless dataset
This repository contains the code associated with our report for the McGill COMP 545 class.

## Dataset Audit
...

## Model Training
All models were trained using the [reference implementation](https://github.com/eric-mitchell/direct-preference-optimization) of DPO. Specifically, we forked their repository and followed the instructions in their README to train the Pythia 2.8B and GPT-2 large models.

## Model Evaluation
Most of the code for generating model is in `inference.py`. See `analysis.ipynb` for our results on the XSTest dataset.
