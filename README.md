Implemntation of speaker verification models like xvector, ecapp-tdnn to train on the [barkopedia invidial dog dataset](https://huggingface.co/spaces/ArlingtonCL2/BarkopediaIndividualDogRecognition). A full analysis of the vocalization part. Dataset can be downloaded [here](https://www.kaggle.com/datasets/nikitasolonitsyn/barkopedia). 
## Installation
```
conda create -n dog_verification python=3.10 -y
conda activate dog_verification
pip install uv
uv sync
```

## Usage
Train the model:
```
uv run train.py HYDRA_ARGS
```

Run inference and generate 3D embedding visualization:
```
uv run inference.py HYDRA_ARGS
```