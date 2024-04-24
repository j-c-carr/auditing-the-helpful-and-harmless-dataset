# Auditing the Helpful and Harmless dataset
This repository contains the code associated with our report for the McGill COMP 545 class.

## Dataset Audit
...

## Model Training
All models were trained using the [reference implementation](https://github.com/eric-mitchell/direct-preference-optimization) of DPO. Specifically, we forked their repository and followed the instructions in their README to train the Pythia 2.8B and GPT-2 large models.

## Model Evaluation
The rest of the files in this repository are for generating and evaluating model outputs on a set of test prompts.
* `inference.py` contains generating model outputs.
* `analysis.ipynb` contains the results from our experiments on XSTest. The `out/xstest_eval` folder contains the generations from our models.
* `inference.py` and `utils.py` help to load and process the datasets.
