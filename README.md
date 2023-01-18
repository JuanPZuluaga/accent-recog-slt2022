# CommonAccent recognition Repository

`Accent Identification from Speech Recordings with ECAPA-TDNN embeddings on CommonAccent`

<p align="center">
    <a href="https://github.com/JuanPZuluaga/accent-recog-slt2022/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-green.svg">
    </a>
    <a href="https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa">
        <img alt="GitHub" src="https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow">
    </a>
    <a href="https://github.com/JuanPZuluaga/accent-recog-slt2022">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Open%20source-green">
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

# Introduction

This is a repository of our submission to [SLT-CODE hackathon](https://slt2022.org/hackathon.php). This hackathon was part of the main SLT-2023 conference: https://slt2022.org/


This repository provides all the necessary tools to perform accent identification from speech recordings with [SpeechBrain](https://github.com/speechbrain/speechbrain) toolkit! The system uses a model fine-tuned on the CommonAccent dataset in English (16 accents). The provided system can recognize the following 16 accents of English from short speech recordings:

```python
<accent-id> <duration-in-hrs>
-----------------------------

african 54.0
australia 196.7
bermuda 7.0
canada 194.3
england 728.6
hongkong 3.4
indian 214.5
ireland 41.3
malaysia 7.1
newzealand 45.5
philippines 7.1
scotland 63.0
singapore 5.0
southatlandtic 2.6
us 1529.9
wales 7.5
```

**Abstract**: The recognition of accented speech still remains a dominant problem in Automatic Speech Recognition (ASR) systems. We approach the classification of accented English speech through the Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA-TDNN) architecture which has been shown to perform well on a variety of speech tasks. Three models are proposed: one trained from scratch, another two models (one using data augmentation and a baseline model) fine-tuned from the checkpoints of speechbrain/spkrec-ecapa-voxceleb (VoxCeleb). Our results show that the model fine-tuned with data augmentation yield the best results. Most of the misclassifications were structured and expected due to accent similarities, such as the American and Canadian accents. We also explored the internal categorization of embeddings through t-SNE, a dimensionality reduction technique, and found that there was a level of clustering based on phonological similarity. For future work, we would like to explore the implementation of this accent classification system in our suggested framework to improve ASR performance by making it more inclusive to accented speech. 



# Get started: 

The first step is to create your environment with the required packages for data preparation, formatting, and to carry out the experiments. You can run the following commands to create the conda environment (assuming CUDA - 11.7):

Step 1: Using python 3.10: install python and the requirements

```bash
git clone https://github.com/JuanPZuluaga/accent-recog-slt2022
conda create -n accent_recog python==3.10
conda activate accent_recog
python -m pip install -r requirements.txt
```


# Database

Our system is trained on the CommonVoice dataset (3.0 version). The portions of data for each set is:

```python
Train set: 50hrs / 45k samples
Dev set: 1.24hrs / 1062 samples
Test set: 1.15hrs / 972 samples
```

Follow the data preparation in: `CommonAccent/common_accent_prepare.py`


# Future work

The results of this project will be submitted to Interspeech 2023. 