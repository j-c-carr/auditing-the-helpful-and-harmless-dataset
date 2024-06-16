#!/bin/bash
#SBATCH --output=logs/job-output-%j_gpt2_rtp_tox_clf.txt
#SBATCH --error=logs/job-error-%j_gpt2_rtp_tox_clf.txt
#SBATCH --mem=64Gb
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:30:00

module load python/3.8
module load cuda/11.7


# Activate the virtual environment (same as used for direct preference optimization)
dpo_dir=$HOME/courses/c597/direct-preference-optimization
source $dpo_dir/venv/bin/activate

set -x

export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub

# Run evaluation script
python toxicity_classification.py 

