# CommonAccent Dataset (CommonVoice 3.0)

This dataset is composed of speakers of 16 different Acents that were carefully selected from [CommonVoice](https://commonvoice.mozilla.org/) database. The total duration of audio recordings is ~53 hours. The data `common_accent_prepare.py` script already splits the data into train, dev (validation) and test sets.

## How to run this script? 

Basically, you need to do the following: 


Step 1: Using python 3.10: install python and the requirements

```python
git clone https://github.com/JuanPZuluaga/accent-recog-slt2022
conda create -n slt_2023 python==3.10
conda activate slt_2023
python -m pip install -r requirements.txt
```

Then, you need to create the manifest files, i.e., CSV files with the train/dev/test sets. You can find an example in `CommonAccent/accent_id/data/train.csv`.

To run the file simply do:

```python
conda activate slt_2023
cd CommonAccent/accent_id/ # run the python script, from this directory
python common_accent_prepare.py /folder/to/commonvoice/data/ data/
```

Info: 
-  `/folder/to/commonvoice/data/`: Folder where you donwloaded CommonVoice, it looks like this (it might be different for you):

```python
├── cv-invalid
├── cv-other-dev
├── cv-other-test
├── cv-other-train
├── cv-valid-dev
├── cv-valid-test
├── cv-valid-train
└── DeepSpeech
```

- `data/`: where to store the manifest files (CSV files).

## Manifest files

The manifest files (train/dev/test.CSV) should look like this:

```python
ID,utt_id,wav,wav_format,duration,accent
0,cv-valid-dev-sample-001066,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001066.mp3,mp3,2.520,england
1,cv-valid-dev-sample-001068,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001068.mp3,mp3,2.664,england
2,cv-valid-dev-sample-001080,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001080.mp3,mp3,6.264,england
3,cv-valid-dev-sample-001127,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001127.mp3,mp3,8.592,england
4,cv-valid-dev-sample-001150,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001150.mp3,mp3,2.544,england
5,cv-valid-dev-sample-001203,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001203.mp3,mp3,2.616,england
6,cv-valid-dev-sample-001206,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001206.mp3,mp3,5.616,england
7,cv-valid-dev-sample-001225,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001225.mp3,mp3,3.864,england
8,cv-valid-dev-sample-001246,/remote/idiap.svm/resource.dataset04/CommonVoice/cv-valid-dev/sample-001246.mp3,mp3,5.640,england
```

It might happen that the samples in the train/dev/test sets are not the same for you (random seed, structure of the dataset, etc)... Anyway, that should be ok, at least for creating a proof of concept system (we can do after some kind of K-fold cross-validation).

## List of Accents:

List of accents of the only-English part of CommonVoice 3.0:

* african
* australia
* bermuda
* canada
* england
* hongkong
* indian
* ireland
* malaysia
* newzealand
* philippines
* scotland
* singapore
* southatlandtic
* us
* wales


## Statistics of CommonAccent (TODO: update):

| Name                              | Train  | Dev    | Test  |
|:---------------------------------:|:------:|:------:|:-----:|
| **# of utterances**               | 45605 | 1062  | 972 |
| **Total duration, hr**            | ~50  | 1.24   | 1.15  |


## Further statistics

Further data can be obtained from the the CommonVoice dataset. Future work should target other languages and also newer versions of the CommonVoice dataset.