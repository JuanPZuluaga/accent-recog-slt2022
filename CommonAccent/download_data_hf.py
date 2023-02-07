
"""
Script to download the CommonVoice dataset using Hugging Face

At Hugging Face: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
Download the dataset: https://commonvoice.mozilla.org/en/datasets

Author
------
Juan Pablo Zuluaga
"""
import os
import argparse
import csv

from datasets import load_dataset, load_from_disk

import warnings
warnings.filterwarnings("ignore")

_COMMON_VOICE_FOLDER = "/remote/idiap.svm/temp.speech01/jzuluaga/experiments/slt/accent-recog-slt2022/CommonAccent/common_voice_11_0/common_voice_11_0.py"

def prepare_cv_from_hf(output_folder, language="en"):
    """ function to prepare the datasets in <output-folder> """

    output_folder = os.path.join(output_folder, language)
    # create the output folder: in case is not present
    os.makedirs(output_folder, exist_ok=True)

    # Prepare the the common voice dataset in streaming mode
    common_voice_ds = load_dataset(_COMMON_VOICE_FOLDER, language, streaming=True)

    # just select relevant splits: train/validation/test set
    splits = ["train", "validation", "test"]
    common_voice = {}
    
    # load, prepare and filter each split in streaming mode:
    for split in splits:
        # filter out samples without accent
        ds = common_voice_ds[split].filter( lambda x: x['accent'] != '')
        common_voice[split] = ds
        
    for dataset in common_voice:
        csv_lines = []
        # Starting index
        idx = 0
        for sample in common_voice[dataset]:
            # get path and utt_id
            mp3_path = sample['path']
            utt_id = mp3_path.split(".")[-2].split("/")[-1]            
            
            # Create a row with metadata + transcripts
            csv_line = [
                idx,  # ID
                utt_id,  # Utterance ID
                mp3_path,  # File name
                sample["locale"],
                sample["accent"],
                sample["age"],
                sample["gender"],
                sample["sentence"], # transcript
            ]

            # Adding this line to the csv_lines list
            csv_lines.append(csv_line)
            # Increment index
            idx += 1

        # CSV column titles
        csv_header = ["idx", "utt_id", "mp3_path", "language", "accent", "age", "gender", "transcript"]
        # Add titles to the list at indexx 0
        csv_lines.insert(0, csv_header)

        # Writing the csv lines
        csv_file = os.path.join(output_folder, dataset+'.tsv')

        with open(csv_file, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_lines:
                csv_writer.writerow(line)
    print(f"Prepare CommonVoice: for {language} in {output_folder}")

def main():
    # read input from CLI, you need to run it from the command lind
    parser = argparse.ArgumentParser()

    # reporting vars
    parser.add_argument(
        "--language",
        type=str,
        default='en',
        help="Language to load",
    )
    parser.add_argument(
        "output_folder",
        help="name of the output folder to store the csv files for each split",
    )
    args = parser.parse_args()

    # call the main function
    prepare_cv_from_hf(output_folder=args.output_folder, language=args.language)

if __name__ == "__main__":
    main()