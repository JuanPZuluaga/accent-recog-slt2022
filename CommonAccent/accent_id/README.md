# Accent Identification experiments with CommonAccent (Common Voice).
This folder contains scripts for running Accent classification experiments with the [CommonVoice dataset](https://commonvoice.mozilla.org/en/datasets) dataset. 

# Training [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
Similar to the X-Vector a bigger and more powerful ECAPA-TDNN model can be used.

`python train.py hparams/train_ecapa_tdnn.yaml`

Data augmentation and environmental corruption are done by concatenating waveforms, dropout, speed change, reverberation, noise, and noise+rev. The batch is double size of the original one. This may lead to
better performance, at the cost of longer training time and higher compute resourses.

# Performance
| Release | hyperparams file | Val. Err | Test Err | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 23-01-08 | train.yaml |  - | - | - |

Each epoch takes approximately 14 minutes on an NVIDIA V100.

# Inference (TODO)
The pre-trained model + easy inference is available on HuggingFace:
- https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa/

Basically, you can run inference with only few lines of code:

```python
import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="pretrained_models/lang-id-commonlanguage_ecapa")

# Italian Example
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-it.wav')
print(text_lab)

# French Example
out_prob, score, index, text_lab = classifier.classify_file('speechbrain/lang-id-commonlanguage_ecapa/example-fr.wav')
print(text_lab)
```


**Web Demo** Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo Audio Classification: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Speechbrain-audio-classification)

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```


