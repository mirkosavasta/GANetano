# GANetano
Generative Adversarial Network to create synthetic time series. This repo refers to a a [Medium Article](https://medium.com/@savastamirko/generating-synthetic-financial-time-series-with-wgans-e03596eb7185) that you should read if you intend to use my code.

## Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/mirkosavasta/GANetano.git
cd GANetano
```
- Install the dependencies
```bash
conda env create -f ganetano_env_reqs.yml
```
### Training
- Train model on sines:
```bash
python3 train.py -ln tensorboard_log_name -ds sines
```
- Train model on arma:
```bash
python3 train.py -ln tensorboard_log_name -ds arma
