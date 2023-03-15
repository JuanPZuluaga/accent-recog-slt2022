#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a XLSR-53 (wav2vec2.0) on Accent Classification for Spanish
#######################################
# COMMAND LINE OPTIONS,
# high-level variables for training the model. TrainingArguments (HuggingFace)
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
set -euo pipefail

# static vars
cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[ij]*' -V'
cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[fghij]*' -V'

# data folder:
csv_prepared_folder="data/es_2k"
output_dir="results/W2V2/ES_small/"
n_accents=6

# training vars
# model from HF hub, it could be another one, e.g., facebook/wav2vec2-base
wav2vec2_hub="facebook/wav2vec2-large-xlsr-53"; encoder_dim=1024
hparams="train_w2v2.yaml"

seed="1986"
apply_augmentation="True"
grad_accumulation_factor=20

# ablation, different learning rates
lr_rates="0.0001 0.0005"
lr_rates=($lr_rates)

for lr_rate in "${lr_rates[@]}"; do
(
    # If augmentation is defined:
    if [ "$apply_augmentation" == "True" ]; then
        output_folder="$output_dir/$(basename $wav2vec2_hub)-augmented/$lr_rate/$seed"
        rir_folder="data/rir_folder/"
        max_batch_len=30
    else
        output_folder="$output_dir/$(basename $wav2vec2_hub)/$lr_rate/$seed"
        rir_folder=""
        max_batch_len=100
    fi

    # configure a GPU to use if we a defined 'CMD'
    if [ ! "$cmd" == 'none' ]; then
        basename=train_$(basename $wav2vec2_hub)_${apply_augmentation}_augmentation_lr-${lr_rate}
        cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
    else
        cmd=''
    fi

    echo "*** About to start the training ***"
    echo "*** output folder: $output_folder ***"
    
    rm -rf ${output_folder}/log/.error
    echo "training model in $output_folder"


    $cmd python3 accent_id/train_w2v2.py accent_id/hparams/$hparams \
        --seed="$seed" \
        --lr_wav2vec2="$lr_rate" \
        --skip_prep="True" \
        --rir_folder="$rir_folder" \
        --n_accents="$n_accents" \
        --number_of_epochs=30 \
        --grad_accumulation_factor=$grad_accumulation_factor \
        --csv_prepared_folder=$csv_prepared_folder \
        --apply_augmentation="$apply_augmentation" \
        --max_batch_len="$max_batch_len" \
        --output_folder="$output_folder" \
        --wav2vec2_hub="$wav2vec2_hub" \
        --encoder_dim="$encoder_dim"


) || touch ${output_folder}/log/.error &
done
wait
if [ -f ${output_folder}/log/.error ]; then
    echo "$0: something went wrong while training the model:"
    echo "$0: ${output_folder}/log/.error"
    exit 1
fi

echo "Done training of $model in $output_folder"
exit 0
