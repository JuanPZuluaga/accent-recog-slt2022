#!/usr/bin/env python3
import os
import sys
import logging

import librosa
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from common_accent_prepare import prepare_common_accent

"""Recipe for performing inference on Accent Classification system with CommonVoice Accent.

To run this recipe, do the following:
> python inference.py hparams/inference_ecapa_tdnn.yaml

Author
------
 * Juan Pablo Zuluaga 2023
"""

logger = logging.getLogger(__name__)


# Brain class for Accent ID training
class AccID_inf(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs
        targets = batch.accent_encoded.data
        loss = self.hparams.compute_cost(predictions, targets)
        
        # Append the outputs here, we can access then later
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)
            self.error_metrics2.append(batch.id, predictions.argmax(dim=-1), targets)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.error_metrics2 = self.hparams.error_stats2()


import ipdb

def dataio_prep(hparams):
    """ This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `common_accent_prepare` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        # sig, _ = torchaudio.load(wav)
        # sig = sig.transpose(0, 1).squeeze(1)
        # Problem with Torchaudio while reading MP3 files (CommonVoice)
        sig, _ = librosa.load(wav, sr=hparams['sample_rate'])
        sig = torch.tensor(sig)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("accent")
    @sb.utils.data_pipeline.provides("accent", "accent_encoded")
    def label_pipeline(accent):
        yield accent
        accent_encoded = accent_encoder.encode_label_torch(accent)
        yield accent_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["dummy", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["csv_prepared_folder"], dataset + ".csv"),
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "accent_encoded"],
        )
        # filtering out recordings with more than max_audio_length allowed    
        datasets[dataset] = datasets[dataset].filtered_sorted(
                key_max_value={"duration": hparams["max_audio_length"]},
        )

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create output directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'accent01': 0, 'accent02': 1, ..)
    accent_encoder = sb.dataio.encoder.CategoricalEncoder()
    
    # Load label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.    
    accent_encoder_file = os.path.join(
        hparams["pretrained_path"], "accent_encoder.txt"
    )
    accent_encoder.load_or_create(
        path=accent_encoder_file,
        output_key="accent",
    )

    # Create dataset objects "train", "dev", and "test" and accent_encoder
    datasets = dataio_prep(hparams)

    # Fetch and laod pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for performing infernence.
    accid_brain = AccID_inf(
        modules=hparams["modules"],
        hparams=hparams,
    )

    # Function that actually prints the output. you can modify this to get some other information
    def print_confusion_matrix(AccID_object, set_name='dev'):
        """ pass the object what contains the stats """

        # get the scores after running the forward pass
        y_true_val = torch.cat(AccID_object.error_metrics2.labels).tolist()
        y_pred_val = torch.cat(AccID_object.error_metrics2.scores).tolist()

        # get the values of the items from the dictionary
        y_true = [accent_encoder.ind2lab[i] for i in y_true_val]
        y_pred = [accent_encoder.ind2lab[i] for i in y_pred_val]

        # retrieve a list of classes
        classes = [i[1] for i in accent_encoder.ind2lab.items()]

        with open(f"{hparams['output_folder']}/classification_report_{set_name}.txt", "w") as f:
            f.write(classification_report(y_true, y_pred))

        # create the confusion matrix and plot it
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        disp.ax_.tick_params(axis='x', labelrotation = 80)
        disp.figure_.savefig(f"{hparams['output_folder']}/conf_mat_{set_name}.png", dpi=300)

    # Load the best checkpoint for evaluation of test set
    test_stats = accid_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
    print_confusion_matrix(accid_brain, set_name="test")

    # Load the best checkpoint for evaluation of dev set
    test_stats = accid_brain.evaluate(
        test_set=datasets["dev"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
    print_confusion_matrix(accid_brain, set_name="dev")

    ipdb.set_trace()
