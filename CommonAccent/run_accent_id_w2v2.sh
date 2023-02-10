#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a XLSR-53 (wav2vec2.0) on Accent Classification for English
#######################################
# COMMAND LINE OPTIONS,
# high-level variables for training the model. TrainingArguments (HuggingFace)
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
set -euo pipefail

# static vars
cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[ij]*' -V'
cmd=none

# training vars

# model from HF hub, it could be another one, e.g., facebook/wav2vec2-base
wav2vec2_hub="facebook/wav2vec2-large-xlsr-53"
seed="1986"
apply_augmentation="True"
max_batch_len=50
batch_size=1

# data folder:
language_id="en"
csv_prepared_folder="data/$language_id"
output_dir="results/W2V2/EN"

# If augmentation is defined:
if [ ! "$apply_augmentation" == 'True' ]; then
    output_folder="$output_dir/$(basename $wav2vec2_hub)-augmented/$seed"
    rir_folder="data/rir_folder/"
else
    output_folder="$output_dir/$(basename $wav2vec2_hub)/$seed"
    rir_folder=""
fi

# configure a GPU to use if we a defined 'CMD'
if [ ! "$cmd" == 'none' ]; then
  basename=train_$(basename $wav2vec2_hub)_${apply_augmentation}_augmentation
  cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
else
  cmd=''
fi

echo "*** About to start the training ***"
echo "*** output folder: $output_folder ***"

$cmd python3 accent_id/train_w2v2.py accent_id/hparams/train_w2v2_xlsr.yaml \
    --seed=$seed \
    --skip_prep="True" \
    --rir_folder="$rir_folder" \
    --csv_prepared_folder=$csv_prepared_folder \
    --apply_augmentation="$apply_augmentation" \
    --max_batch_len="$max_batch_len" --batch_size="$batch_size" \
    --output_folder="$output_folder" \
    --wav2vec2_hub="$wav2vec2_hub" 

echo "Done training of $model in $output_folder"
exit 0