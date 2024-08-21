#!/bin/bash
#SBATCH --output=logs/job-output-%j_gpt_xstest.txt
#SBATCH --error=logs/job-error-%j_gpt_xstest.txt
#SBATCH --mem=64Gb
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:20:00

module load python/3.8
module load cuda/11.7


# Activate the virtual environment (same as used for direct preference optimization)
dpo_dir=$HOME/courses/c597/direct-preference-optimization
source $dpo_dir/venv/bin/activate

# GPT-2 model checkpoints
gpt_checkpoint_dir="/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/v1_checkpoints/gpt2-large"
gpt_help_only="${gpt_checkpoint_dir}/helpful_only/dpo_gpt2l_helpful_longer_2024-04-13_step-200000_policy.pt"
gpt_hh_filtered="${gpt_checkpoint_dir}/all_filtered/gpt2l_dpo_filtered_longer_2024-04-14_step-280000_policy.pt"
gpt_hh_harmless="${gpt_checkpoint_dir}/hh_harmless_harmless/gpt2l_dpo_harmless_harmless_Jun13.pt"
gpt_hh_full="${gpt_checkpoint_dir}/hh_full/dpo_gpt2l_paper_params_longer_2024-04-13_step-240000_policy.pt"

# Pythia model checkpoints
pythia_checkpoint_dir="/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/v1_checkpoints/pythia28"
pythia_hh_harmless="${pythia_checkpoint_dir}/hh_harmless_harmless/hh_dpo_pythia28_harmless_harmless_Jun14_1epoch.pt"
pythia_help_only="${pythia_checkpoint_dir}/helpful_only/dpo_pythia28_helpful_only_2024_04_16_step-160000.pt"
pythia_hh_filtered="${pythia_checkpoint_dir}/all_filtered/dpo_pythia28_filtered_2024-04-16_step-160000_policy.pt"
pythia_hh_full="${pythia_checkpoint_dir}/hh_full/dpo_pythia28_hh_full_1_epoch.pt"

# Mistral model checkpoints
mistral_help_only="/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4804526/final_checkpoint"
mistral_hh_full_dpo="/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4796852/final_checkpoint"

# Llama model checkpoints
llama_help_only_dpo="/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4807943/final_checkpoint"
llama_hh_full_dpo="/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4802734/final_checkpoint"

set -x

# Dataset directories
export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
xstest_dset_dir='/network/scratch/j/jonathan.colaco-carr/hh_fruits/data/xstest'

# Run inference
python run_inference.py \
  --base_model_name="gpt2-large" \
  --model_checkpoint=$gpt_hh_full \
  --model_name="hh_full" \
  --dset_name="xstest-plus" \
  --dset_dir=$xstest_dset_dir \
  --batch_size=32 \
  --num_return_sequences=3 \
  --top_p=0.95 \
  --top_k=50 \
  --max_new_tokens=50 \
  --do_sample


