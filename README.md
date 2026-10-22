DeepFormants
============

Shua Dissen (shua.dissen@gmail.com)            
Joseph Keshet (joseph.keshet@biu.ac.il)  


DeepFormants is a software package for formant tracking and estimation, using deep networks. It works as follows:
* The user provides a wav file with an initial stop consonant. 
* Optionally, the user can designate a window for estimation by providing a start and end time (specified in seconds).
* A classifer is used to estimate the formants in the file, with two modes:
* Estimation: If a time window is specified, a single estimate is made for F1-F4 within that window. 
* Tracking: If no time window is given, the model will track F1-F4 and give a measurement at every 10 milliseconds across the length of the file.

Four models are available, two per task, selected with `--backend`:

| Task | `--backend` | Model | Input |
|---|---|---|---|
| Estimation | `ff` (default) | dense net | 350-dim LPC/cepstral vector |
| Estimation | `cnn` | 4-layer CNN | 55x50 spectrogram image |
| Tracking | `rnn` (default) | 2-layer LSTM | 350-dim LPC/cepstral vector, per frame |
| Tracking | `crnn` | ConvLSTM2D + conv tower | 55x50 spectrogram patches, per frame |

This is a beta version of DeepFormants. Any reports of bugs, comments on how to improve the software or documentation, or questions are greatly appreciated, and should be sent to the authors at the addresses given above.

---


## Installation instructions

Download the code. Estimation and tracking both run on [PyTorch](https://pytorch.org); the
original Torch7/Lua loaders are no longer needed. LPC analysis is done by the bundled
`levinson_lpc.py`, so `scikits.talkbox` is no longer a dependency either.

Dependencies:
```
pip install -r requirements.txt
```
That installs `torch`, `numpy` and `scipy`, which is everything the four models need for
both training and inference. `requirements.txt` also lists three optional extras, commented
out: `h5py` (only to re-convert the original Keras CRNN weights), `huggingface_hub` (only
as a fallback for a checkpoint that is not committed) and `wandb` (only for the `--wandb`
training flag).

`sox` is used to resample input audio to 16 kHz mono and is not installable with pip:
```
sudo apt-get install sox
```

All four sets of weights are committed under `pytorchFormants/` and load automatically:

| Backend | Checkpoint | Size |
|---|---|---|
| `ff` | `Estimator/LPC_NN_scaledLoss.pt` | 3.9 MB |
| `cnn` | `Estimator/CNN_estimate.pt` | 16.8 MB |
| `rnn` | `Tracker/LPC_RNN.pt` | 9.8 MB |
| `crnn` | `CRNN/CRNN_tracker.pt` | 0.8 MB |

`pytorchFormants/weights.py` resolves checkpoints: an explicit `--model` wins, otherwise a
committed copy is used, and only if one is missing is it fetched from Hugging Face
(set `DEEPFORMANTS_HF_REPO` to override the repo). Install `huggingface_hub` if you rely
on that fallback.

The legacy Torch7 files (`estimation_model.dat`, `load_estimation_model.lua`,
`load_tracking_model.lua`) are kept only for reference.

## How to use:

For vowel formant estimation, call the main script in a terminal with the following inputs: wav file, formant output filename, and the vowel begin and end times:

```
python formants.py data/Example.wav data/ExamplePredictions.csv --begin 1.2 --end 1.3
```

or the vowel begin and end times can be taken from a TextGrid file (here the name of the TextGrid is Example.TextGrid and the vowel is taken from a tier called "VOWEL"):

```
python formants.py data/Example.wav data/examplePredictions.csv --textgrid_filename data/Example.TextGrid \
          --textgrid_tier VOWEL
```

`data/Example.TextGrid` marks two windows for demonstration; it is not a phonetic
annotation.

For formant tracking, just call the script with the wav file and output filename:

```
python formants.py data/Example.wav data/ExamplePredictions.csv
```

### Running each of the four models

Estimation accepts `ff` or `cnn`, tracking accepts `rnn` or `crnn`, and a backend that does
not match the task is rejected rather than silently ignored. Each of these runs as-is on
the bundled example file:

```
# Estimation - one F1-F4 for the 1.2-1.3s window
python formants.py data/Example.wav preds_ff.csv   --begin 1.2 --end 1.3 --backend ff
python formants.py data/Example.wav preds_cnn.csv  --begin 1.2 --end 1.3 --backend cnn

# Tracking - F1-F4 every 10ms, 231 frames for this file
python formants.py data/Example.wav preds_rnn.csv  --backend rnn
python formants.py data/Example.wav preds_crnn.csv --backend crnn
```

Omitting `--backend` gives `ff` for estimation and `rnn` for tracking. Expected output on
the example file, in Hz:

```
                  F1      F2      F3      F4
ff             458.5  1642.7  2510.4  3689.1
cnn            418.2  1733.6  2522.0  3614.4
rnn    (frame 0)  659.4  1763.5  2798.1  4080.7
crnn   (frame 0)  580.8  1603.5  2507.5  3914.7
```

To use the retrained CRNN instead of the converted original:

```
python formants.py data/Example.wav preds.csv --backend crnn \
          --model pytorchFormants/CRNN/CRNN_tracker_retrained.pt
```


## Evaluation

`evaluate.py` scores a model on a prepared test split, reporting per-formant mean
absolute error in Hz. Add `--label_root` for the per-phone-class breakdown:

```
python evaluate.py --backend rnn --data .../Outputs/Test_tracker.npy \
          --label_root .../shua_processed/Test
```

Two CRNN checkpoints are shipped. `CRNN_tracker.pt` is converted from the original Keras
weights and is the default. `CRNN_tracker_retrained.pt` was trained from scratch with
this code and performs equivalently; select it with `--model`.


## Training

### A note on the data

The corpora used here are licensed to us and are **not ours to redistribute**: VTR
annotations over TIMIT audio (LDC), Hillenbrand, and Clopper. Neither the audio nor any
features derived from it are published with this repository, and `.gitignore` is set up to
keep derived `.npy` files out of commits. Obtain the corpora yourself, then build the
features locally with the scripts below.

The dataset paths shown are the defaults on our machines; pass your own with the
corresponding flags.

Each model has a matching `*_train.py` that takes its dataset paths as arguments, reports
per-formant test error in Hz, and writes a checkpoint only when the model improves. Add
`--wandb` to any of them to log to Weights & Biases.

Labels of `-1` mark an unannotated formant (Clopper annotates only F1/F2, Hillenbrand only
a few points per vowel). Every training script masks those out of the loss rather than
regressing towards -1 kHz.

### Feature building

The LPC/cepstral features for `ff` and `rnn`:

```
python pytorchFormants/Estimator/create_VTR_feats.py     # -> Train.npy, Test.npy
python pytorchFormants/Tracker/create_VTR_feats.py       # -> Train_tracker.npy, Test_tracker.npy
```

The spectrogram images for `cnn`, optionally merging corpora:

```
python pytorchFormants/Estimator/build_cnn_data.py --corpora vtr hillenbrand clopper \
          --prefix merged --output_dir data/cnn/
```

The spectrogram patch sequences for `crnn`. Expects a directory tree of WAVs each with a
matching `.label` file (tab separated: phone, then F1-F4 in Hz, one row per 10 ms):

```
python pytorchFormants/CRNN/crnn_feats.py --train_dir /path/Train/ --test_dir /path/Test/ \
          --output_dir data/crnn/
```

### Training runs

```
python pytorchFormants/Estimator/ff_train.py   --train_data .../Train.npy \
          --test_data .../Test.npy

python pytorchFormants/Estimator/cnn_train.py  --train_X data/cnn/merged_train_X.npy \
          --train_Y data/cnn/merged_train_Y.npy --test_X data/cnn/merged_test_X.npy \
          --test_Y data/cnn/merged_test_Y.npy --arch keras

python pytorchFormants/Tracker/rnn_train.py    --train_data .../Train_tracker.npy \
          --test_data .../Test_tracker.npy

python pytorchFormants/CRNN/crnn_train.py      --train_data data/crnn/Train_crnn.npy \
          --test_data data/crnn/Test_crnn.npy
```

`cnn_train.py --arch` picks the topology: `keras` reproduces the original estimator and is
the default; `legacy` is a larger variant whose first dense layer alone holds 53.5M
parameters. Checkpoints are routed to the right topology automatically, so either loads
without a flag.


## The CRNN tracker

The CRNN is a ConvLSTM2D over spectrogram patches: the recurrence lives inside the
convolution rather than after it.

```
input (batch, 10, 1, 55, 50)
ConvLSTM2D(64, 3x3, same, relu / hard_sigmoid, return_sequences)
TimeDistributed[ Conv2D(32,3x3) -> MaxPool(2,2) ] x4          # 55x50 -> 1x1
TimeDistributed[ Flatten -> Dropout -> Dense(256) -> Dropout -> Dense(4) ]
```

The committed weights are converted from the original Keras 1.2.2 model. Theano convolves
where PyTorch cross-correlates, so the conversion flips every conv kernel on both spatial
axes; the result reproduces the original model's own predictions to 0.002 Hz. To redo the
conversion from a `.h5`:

```
python pytorchFormants/CRNN/convert_keras_weights.py weights_best2.h5 \
          pytorchFormants/CRNN/CRNN_tracker.pt
```


