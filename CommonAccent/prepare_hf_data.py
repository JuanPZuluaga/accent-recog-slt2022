
"""
Script to download the CommonVoice dataset using Hugging Face

At Hugging Face: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
Download the dataset: https://commonvoice.mozilla.org/en/datasets

Author
------
Juan Pablo Zuluaga
"""
import os

from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")

_ACCENTS = [
    "african",
    "australia",
    "bermuda",
    "canada",
    "england",
    "hongkong",
    "indian",
    "ireland",
    "malaysia",
    "newzealand",
    "philippines",
    "scotland",
    "singapore",
    "southatlandtic",
    "us",
    "wales",
]
_COMMON_VOICE_VERSION = "mozilla-foundation/common_voice_11_0"
_LANGUAGE = "en"
_CACHE_DIR = "/remote/idiap.svm/temp.speech01/jzuluaga/experiments/slt/data/cv_11"

import ipdb

def main():

    # get number of CPUs
    nproc = os.cpu_count()

    ipdb.set_trace()
    # Download the common voice dataset
    common_voice = load_dataset(_COMMON_VOICE_VERSION, _LANGUAGE, cache_dir=_CACHE_DIR, num_proc=nproc-2)

    print(f"CommonVoice: {_COMMON_VOICE_VERSION} for {_LANGUAGE}")
    print(f"Saved here: {_CACHE_DIR}")

if __name__ == "__main__":
    main()
