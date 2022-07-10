# Niri-Zero's Computer Vision

## About the project
This repository contains the code the team used to detect empty spaces on shelves with bounding boxes.

## Running Locally

- Install all required packages from requirements.txt
- Download the training files here: https://drive.google.com/drive/folders/1fXy5brt6oLXJ79CdnXkvat4jCT0BNsJy?usp=sharing
- All training files are stored as JSONs in Label-Me format (image data is seralized inside the Json)

Install the dependencies and devDependencies and start the server.

## Architecture
For our project, we chose to use a Swin-T Transformer Backbone + Faster RCNN head to try and extract more context form each image patch. https://arxiv.org/abs/2103.14030
https://arxiv.org/pdf/1506.01497.pdf

We use AdamW as our optimizer alongside an aggressive OneCycleLearning Scheduler. This allows us to specify larger learning rates and lead to superconvergence. https://arxiv.org/pdf/1803.09820.pdf

| Hyperparameters | Value |
| ------ | ------ |
| Seed | 42 |
| Epochs | 8 |
| Folds | 5 |
| Max Learning Rate | 1e-2 |
| Weight Decay | 0.5 |


