# Welcome! 
This repository is about taking on the Kaggle challenge 
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview).
It currently makes use of models available in `torchtext` and has achieved the following accuracy scores on the Kaggle leaderboard

1. 83.726 - RoBERTa encoder with large configuration trained for 1 epoch
2. 78.700 - XLM-R encoder with base configuration trained for 10 epochs
3. 74.931 - XLM-R encoder with base configuration trained for 1 epoch

# Getting started
## Install dependencies
This project's requirements can be installed with
`pip install -r requirements.txt`

To accelerate training on a CUDA-enabled machine, you can [get started with CUDA locally](https://pytorch.org/get-started/locally/) 

Note that depending on your setup, you may need to add the `--upgrade --force-reinstall` flags when installing CUDA-enabled torch from pip

## Download the datasets
1. Create a Kaggle account
2. Join the competition: Natural Language Processing with Disaster Tweets
3. Install the kaggle command line tool: https://www.kaggle.com/docs/api
    - Make sure you generate an API token as described in the authentication section
4. Run `kaggle competitions download -c nlp-getting-started`
5. (Optional) to follow the examples in this repository more closely, put the downloaded data in a directory called "data"

## Check out the examples
See `examples.py` for actual usage of this repository.

Pretrained model parameters will be uploaded soon to get you going more quickly!