# CommonAccent (CV 11.0) recognition Repository

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

**Abstract**: The recognition of accented speech still remains a dominant problem in Automatic Speech Recognition (ASR) systems. We approach the classification of accented English speech through the Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA-TDNN) architecture which has been shown to perform well on a variety of speech tasks. Three models are proposed: one trained from scratch, another two models (one using data augmentation and a baseline model) fine-tuned from the checkpoints of speechbrain/spkrec-ecapa-voxceleb (VoxCeleb). Our results show that the model fine-tuned with data augmentation yield the best results. Most of the misclassifications were structured and expected due to accent similarities, such as the American and Canadian accents. We also explored the internal categorization of embeddings through t-SNE, a dimensionality reduction technique, and found that there was a level of clustering based on phonological similarity. For future work, we would like to explore the implementation of this accent classification system in our suggested framework to improve ASR performance by making it more inclusive to accented speech. 

This repository provides all the necessary tools to perform accent identification from speech recordings with [SpeechBrain](https://github.com/speechbrain/speechbrain) toolkit! The system uses a model fine-tuned on the CommonAccent dataset in English (21 accents). The provided system can recognize the following 21 accents of English from short speech recordings:


```python
<accent-id> <number-of-samples-in-cv>
-----------------------------
* Austrian - 104
* East African Khoja - 107
* Dutch - 108
* West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad) - 282
* Welsh English - 623
* Malaysian English - 1004
* Liverpool English,Lancashire English,England English - 2571
* Singaporean English - 2792
* Hong Kong English - 2951
* Filipino - 4030
* Southern African (South Africa, Zimbabwe, Namibia) - 4270
* New Zealand English - 4960
* Irish English - 6339
* Northern Irish - 6862
* Scottish English - 10817
* Australian English - 33335
* German English,Non native speaker - 41258
* Canadian English - 45640
* England English - 75772
* India and South Asia (India, Pakistan, Sri Lanka) - 79043
* United States English - 249284
```

We also have developed AccentID system for the following languages:

```python
<accent-id> <duration-in-hrs>
-----------------------------
SPANISH (ES)
* España: Islas Canarias - 1326
* Chileno: Chile, Cuyo - 4285
* América central - 6031
* Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México - 8329
* España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha) - 8683
* Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay - 11162
* Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina - 12997
* México - 26136
* España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca) - 30588
* España: Sur peninsular (Andalucia, Extremadura, Murcia) - 38251

FRENCH (FR):
* Français d’Algérie - 319 
* Français d’Allemagne - 355 
* Français du Bénin - 823 
* Français de La Réunion - 884 
* Français des États-Unis - 898 
* Français de Suisse - 3608 
* Français de Belgique - 6509 
* Français du Canada - 8073 
* Français de France - 342921 

GERMAN (DE):
* Italienisch Deutsch - 947 
* Schweizerdeutsch - 9891 
* Österreichisches Deutsch - 16066 
* Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch - 50843 
* Deutschland Deutsch - 252709 

ITALIAN (IT)
* Emiliano - 151
* Meridionale - 193
* Veneto - 1508
* Tendente al siculo, ma non marcato - 2175
* Basilicata,trentino - 2297
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

You need to run this to get Pytorch running with CUDA 11.6

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

# Database (TO update)

Our system is trained on the CommonVoice dataset (11.0 version). Follow the data preparation (`CommonAccent/common_accent_prepare.py`) in `CommonAccent/README.md`

# Future work

The results of this project will be submitted to Interspeech 2023. 