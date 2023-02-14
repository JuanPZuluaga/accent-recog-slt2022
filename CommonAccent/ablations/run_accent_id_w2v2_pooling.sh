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

# Kill sub-processes on exit,
# This command kills any child process generated here
trap "pkill -P $$" EXIT SIGINT SIGTERM

# static vars
cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[ij]*' -V'

# training vars

# model from HF hub, it could be another one, e.g., facebook/wav2vec2-base
wav2vec2_hub="facebook/wav2vec2-large-xlsr-53"; hparams="train_w2v2_xlsr.yaml"
wav2vec2_hub="facebook/wav2vec2-base"; hparams="train_w2v2.yaml"
seed="1986"
apply_augmentation="False"
n_accents=21

# data folder:
csv_prepared_folder="data/en"
output_dir="results/W2V2/EN"

echo "*** About to start the Pooling Strategy Ablation ***"

# ablation, pooling strategy
pooling_strategies="statpool adaptivepool avgpool"
pooling_strategies=($pooling_strategies)

for pooling_strategy in "${pooling_strategies[@]}"; do
(
    # If augmentation is defined:
    if [ "$apply_augmentation" == "True" ]; then
        output_folder="$output_dir/$(basename $wav2vec2_hub)-augmented-$pooling_strategy/$seed"
        rir_folder="data/rir_folder/"
        max_batch_len=300
    else
        output_folder="$output_dir/$(basename $wav2vec2_hub)-$pooling_strategy/$seed"
        rir_folder=""
        max_batch_len=600
    fi

    # configure a GPU to use if we a defined 'CMD'
    if [ ! "$cmd" == 'none' ]; then
        basename=train_$(basename $wav2vec2_hub)_${apply_augmentation}_augmentation_$pooling_strategy
        cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
    else
        cmd=''
    fi
    rm -rf ${output_folder}/.error
    echo "training model in $output_folder"

    # running the training
    $cmd python3 accent_id/train_w2v2.py accent_id/hparams/$hparams \
        --seed="$seed" \
        --skip_prep="True" \
        --rir_folder="$rir_folder" \
        --n_accents="$n_accents" \
        --csv_prepared_folder="$csv_prepared_folder" \
        --apply_augmentation="$apply_augmentation" \
        --max_batch_len="$max_batch_len" \
        --output_folder="$output_folder" \
        --wav2vec2_hub="$wav2vec2_hub" \
        --avg_pool_class="$pooling_strategy"

) || touch ${output_folder}/.error &
done
wait
if [ -f ${output_folder}/.error ]; then
    echo "$0: something went wrong while training the model:"
    echo "$0: ${output_folder}/.error"
    exit 1
fi

echo "Done training of $wav2vec2_hub in $output_folder"
exit 0
