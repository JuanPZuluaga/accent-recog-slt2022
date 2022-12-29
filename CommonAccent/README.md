# CommonAccent Dataset (CommonVoice 3.0)

This dataset is composed of speakers of 12 different Acents that were carefully selected from [CommonVoice](https://commonvoice.mozilla.org/) database. The total duration of audio recordings is XXX hours. The data `common_accent_prepare.py` script already splits the data into train, dev (validation) and test sets.

## How to run this script? 

Basically, you need to do the following: 


Step 1: Using python 3.10: install python and the requirements

```python
git clone https://github.com/JuanPZuluaga/accent-recog-slt2022
conda create -n slt_2023 python==3.10
conda activate slt_2023
python -m pip install -r requirements.txt
```

Then, you need to create the manifest files, which basically are CSV files with the train/dev/test sets. You can find an example in `CommonAccent/accent_id/data/train.csv`.

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


## List of languages:
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


## Statistics of CommonAccent (TO update):

| Name                              | Train  | Dev    | Test  |
|:---------------------------------:|:------:|:------:|:-----:|
| **# of utterances**               | 177552 | 47104  | 47704 |
| **# unique speakers**             | 11189  | 1297   | 1322  |
| **Total duration, hr**            | 30.04  | 7.53   | 7.53  |
| **Min duration, sec**             | 0.86   | 0.98   | 0.89  |
| **Mean duration, sec**            | 4.87   | 4.61   | 4.55  |
| **Max duration, sec**             | 21.72  | 105.67 | 29.83 |
| **Duration per language, min**    | ~40    | ~10    | ~10   |


## Other information (TODO)
In addition to the language label, the datapoints have `age`, `gender` and `utterance transcription` associated with each utterance.
