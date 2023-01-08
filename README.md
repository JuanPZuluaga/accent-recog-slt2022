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

This is a repository of our submission to [SLT-CODE hackathon](https://slt2022.org/hackathon.php). this hackathon was part of the main SLT-2023 conference: https://slt2022.org/


This repository provides all the necessary tools to perform accent identification from speech recordings with [SpeechBrain](https://github.com/speechbrain/speechbrain) toolkit! The system uses a model fine-tuned on the CommonAccent dataset in English (16 accents). The provided system can recognize the following 16 accents of English from short speech recordings:

```python
african
australia
bermuda
canada
england
hongkong
indian
ireland
malaysia
newzealand
philippines
scotland
singapore
southatlandtic
us
wales
```


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