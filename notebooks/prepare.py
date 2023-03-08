import torch
from speechbrain.pretrained import Pretrained

class WaveformEncoder(Pretrained):
    """A ready-to-use waveformEncoder model

    It can be used to wrap different embedding models such as SSL ones (wav2vec2)
    or speaker ones (Xvector) etc. Two functions are available: encode_batch and
    encode_file. They can be used to obtain the embeddings directly from an audio
    file or from a batch of audio tensors respectively.

    The given YAML must contain the fields specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import WaveformEncoder
    >>> tmpdir = getfixture("tmpdir")
    >>> ssl_model = WaveformEncoder.from_hparams(
    ...     source="speechbrain/ssl-wav2vec2-base-libri",
    ...     savedir=tmpdir,
    ... ) # doctest: +SKIP
    >>> ssl_model.encode_file("samples/audio_samples/example_fr.wav") # doctest: +SKIP
    """

    MODULES_NEEDED = ["encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_file(self, path):
        """Encode the given audiofile into a sequence of embeddings.

        Arguments
        ---------
        path : str
            Path to audio file which to encode.

        Returns
        -------
        torch.Tensor
            The audiofile embeddings produced by this system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        results = self.encode_batch(batch, rel_length)
        return results["embeddings"]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def forward(self, wavs, wav_lens):
        """Runs the encoder"""
        return self.encode_batch(wavs, wav_lens)
