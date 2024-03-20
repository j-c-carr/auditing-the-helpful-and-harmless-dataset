# hh_lhf_inference
## Instructions
### 1. Download the code and create a virtual environment on the Mila cluster 
```
# Log in to the Mila cluster and clone this repository
ssh mila
git clone https://github.com/j-c-carr/hh_lhf_inference.git

# Create an interactive session to test the model
salloc --gres=gpu:1 --mem=32Gb --time=1:00:00

# Create the virtual environment using python 3.8 and cuda 11.7
module load python/3.8
module load cuda/11.7

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run `inference.py`
```
python inference.py
```
