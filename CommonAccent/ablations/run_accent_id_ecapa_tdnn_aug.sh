#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a ECAPA-TDNN model on Accent Classification for English
#######################################
# COMMAND LINE OPTIONS,
# high-level variables for training the model. TrainingArguments (HuggingFace)
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
set -euo pipefail

# static vars
cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[ij]*' -V'

# training vars
ecapa_tdnn_hub="speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt"
seed="1986"
apply_augmentation="True"
max_batch_len=150
n_accents=14

# data folder:
csv_prepared_folder="data/en"
output_dir="results/ECAPA-TDNN/EN/spkrec-ecapa-voxceleb"

# If augmentation is defined:
if [ "$apply_augmentation" == "True" ]; then
    output_folder="${output_dir}-augmented/$seed"
    rir_folder="data/rir_folder/"
else
    output_folder="$output_dir/$seed"
    rir_folder=""
fi

# configure a GPU to use if we a defined 'CMD'
if [ ! "$cmd" == 'none' ]; then
  basename=train_$(basename $ecapa_tdnn_hub)_${apply_augmentation}_augmentation
  cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
else
  cmd=''
fi

echo "*** About to start the training ***"
echo "*** output folder: $output_folder ***"

$cmd python accent_id/train.py accent_id/hparams/train_ecapa_tdnn.yaml \
    --seed=$seed \
    --skip_prep="True" \
    --rir_folder="$rir_folder" \
    --csv_prepared_folder=$csv_prepared_folder \
    --apply_augmentation="$apply_augmentation" \
    --max_batch_len="$max_batch_len" \
    --output_folder="$output_folder" \
    --ecapa_tdnn_hub="$ecapa_tdnn_hub" \
    --n_accents=$n_accents

echo "Done training of $ecapa_tdnn_hub in $output_folder"
exit 0