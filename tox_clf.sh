#!/bin/bash
#SBATCH --output=logs/job-output-%j_hh_harmless_tox_clf.txt
#SBATCH --error=logs/job-error-%j_hh_harmless_clf.txt
#SBATCH --mem=32Gb
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --mail-user=jonathan.colaco-carr@mila.quebec

module load python/3.8
module load cuda/11.7


# Activate the virtual environment (same as used for direct preference optimization)
dpo_dir=$HOME/courses/c597/direct-preference-optimization
source $dpo_dir/venv/bin/activate

set -x

export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub

hh_full=out/rtp_eval/hh_full_Cor_EleutherAI-pythia-2.8b_rtp_3_sequences.csv
hh_filtered=out/rtp_eval/hh_filtered_Cor_EleutherAI-pythia-2.8b_rtp_3_sequences.csv
help_only=out/rtp_eval/help_only_Cor_EleutherAI-pythia-2.8b_rtp_3_sequences.csv
hh_harmless=out/rtp_eval/hh_harmless_Cor_EleutherAI-pythia-2.8b_rtp_3_sequences.csv
hh_rlhf_tmp=out/rtp_eval/hh_full_Cor_pythia_rlhf_rtp_3_sequences.csv

# Run evaluation script
python3 toxicity_classification.py $hh_rlhf_tmp 

