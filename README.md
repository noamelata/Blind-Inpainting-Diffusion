
<h1 align="center">VOutliers Detection and Removal Using Diffusion Models Combined with Confidence Intervals</h1>
<h2 align="center">Reliability, Equity, and Reproducibility in Modern Machine Learning - 048100
</h2> 

  <p align="center">
    Noam Elata: <a href="https://github.com/noamelata">GitHub</a>
  <br>
    Shahar Yadin: <a href="https://github.com/shaharYadin">GitHub</a>
  </p>


## Background
In this project we equip diffusion models with confidence intervals as done in the paper <a href="https://arxiv.org/pdf/2211.09795.pdf">CONffusion: confidence intervals for diffusion models"</a>. 
We demonstrate that using the confidence intervals, we can solve the task of blind image inpainting, removing simple image artifacts.

## Preparation
Download the <a href="https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256">CelebA-HQ256</a> dataset and the Diffusion Model checkpoint from <a href="https://github.com/ermongroup/SDEdit">SDEdit</a>.

Create a conda environment using `environment.yml`.

### Training

Finetune the pretrained model using the training script for the desired quantiles:
```
python train.py -c configs/celeba_hq_q005.json

python train.py -c configs/celeba_hq_q095.json
```

### Calibrate the models using

Calibrate using the calibration set:

```
python calibrate.py -c configs/celeba_hq_q005.json \
                    -ml <lower quantile model checkpoint> -mh <higher quantile model checkpoint>
```

### Inference & Visual Results

Run the following script:
```
python calibrate.py -c configs/celeba_hq_q005.json \
                    -ml <lower quantile model checkpoint> -mh <higher quantile model checkpoint> \
                    --artifact ["red", "rainbow", "butterflys"]
```

### Refrences
* [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733). 
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
