#!/bin/bash
#SBATCH --output=logs/job-output-%j_pythia28_hh_full_2_epochs_xstest.txt
#SBATCH --error=logs/job-error-%j_pythia28_hh_full_2_epochs_xstest.txt
#SBATCH --mem=64Gb
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --time=5:30:00

module load python/3.8
module load cuda/11.7


# Activate the virtual environment
source $HOME/hh_lhf_training/venv/bin/activate

# For smaller models, use --mem=64Gb -cpus-per-gpu=8 --gres:gpu:1
# For bigger models, use --mem=64Gb --cpus-per-gpu=8 --gres=gpu:2

checkpoint_dir="$SCRATCH/hh_fruits/checkpoints/v1_checkpoints"

# Pythia28 models
pythia28_hh_full_1_epoch="$checkpoint_dir/pythia28/hh_full/pythia28_dpo_hh_full_1_epoch_5391350"
pythia28_hh_full_2_epoch="$checkpoint_dir/pythia28/hh_full/pythia28_dpo_hh_full_2_epochs_5401460"
pythia28_hh_filtered_1_epoch="$checkpoint_dir/pythia28/hh_filtered_v2/pythia_28_hh_filtered_1_epoch_5358431"
pythia28_hh_filtered_2_epoch="$checkpoint_dir/pythia28/hh_filtered_v2/pythia_28_hh_filtered_2_epochs_5407249"
pythia28_help_only_1_epoch="$checkpoint_dir/pythia28/helpful_only/pythia_28_dpo_help_only_1_epoch_5398290"
pythia28_help_only_2_epoch="$checkpoint_dir/pythia28/helpful_only/pythia_28_help_only_2_epochs_5399975"

# GPT 2L models
gpt2l_hh_full_1_epoch="$checkpoint_dir/gpt2-large/hh_full/hh_full_1_epoch_5391344"
gpt2l_hh_full_2_epoch="$checkpoint_dir/gpt2-large/hh_full/hh_full_2_epochs_5401459"
gpt2l_hh_filtered_1_epoch="$checkpoint_dir/gpt2-large/hh_filtered_v2/hh_filtered_1_epoch_5358427"
gpt2l_hh_filtered_2_epoch="$checkpoint_dir/gpt2-large/hh_filtered_v2/hh_filtered_2_epochs_5407253"
gpt2l_help_only_1_epoch="$checkpoint_dir/gpt2-large/helpful_only/help_only_1_epoch_5398287"
gpt2l_help_only_2_epoch="$checkpoint_dir/gpt2-large/helpful_only/help_only_2_epochs_5399958"

# OPT 2.7 models
opt27_hh_full_2_epoch="$checkpoint_dir/opt27/hh_full/hh_full_2_epochs_5417691"
opt27_hh_filtered_2_epoch="$checkpoint_dir/opt27/hh_filtered_v2/hh_filtered_2_epochs_5417692"
opt27_help_only_2_epoch="$checkpoint_dir/opt27/helpful_only/help_only_2_epochs_5417652"

# Hugging face access token is required for Mistral and Llama models
export HF_TOKEN=$(<.hf_access_token)

set -x

# Dataset directories
export HF_HOME=$SCRATCH/cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
xstest_dset_dir='/network/scratch/j/jonathan.colaco-carr/hh_fruits/data/xstest'

# Run inference
python run_inference_with_vllm.py \
  --base_model_name="pythia28" \
  --model_checkpoint=$pythia28_hh_full_2_epoch\
  --model_name="pythia28_hh_full_2_epoch" \
  --dset_name="xstest" \
  --dset_dir=$xstest_dset_dir \
  --batch_size=128 \
  --num_return_sequences=25 \
  --top_p=0.9 \
  --temperature=1.0 \
  --max_new_tokens=20


