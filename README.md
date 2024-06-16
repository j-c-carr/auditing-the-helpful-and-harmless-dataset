# Auditing the Helpful and Harmless dataset
This repository contains the code associated with our manuscript, "Beyond the Safety Bundle: Auditing the Helpful and Harmless dataset"

## Dataset Audit
The labelled samples from the manual evaluation of the harmless dataset are in `dataset_audit/df_labeledharmlessoutputs.csv`

## Model Training
All models were trained using the [reference implementation](https://github.com/eric-mitchell/direct-preference-optimization) of DPO. Specifically, we forked their repository and followed the instructions in their README to train the Pythia 2.8B and GPT-2 large models.

## Model Evaluation
The rest of the files in this repository are for generating and evaluating model outputs on a set of test prompts.
* `main.py` contains the main code to generate model outputs.
* `toxicity_classification.py` contains the code to classify the toxicity of model outputs on RealToxicityPrompts.
* `analysis.ipynb` contains the results from our experiments on XSTest. The `out/xstest_eval` folder contains the generations from our models.
* `inference_datasets.py` and `utils.py` help to load and process the datasets.
