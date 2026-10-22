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
pip install torch numpy scipy
```
`sox` is used to resample input audio to 16 kHz mono:
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

For formant tracking, just call the script with the wav file and output filename:

```
python formants.py data/Example.wav data/ExamplePredictions.csv
```

This uses the LSTM tracker over LPC features (`--backend rnn`, the default).

To use a different model, pass `--backend`:

```
python formants.py data/Example.wav preds.csv --begin 1.2 --end 1.3 --backend cnn
python formants.py data/Example.wav preds.csv --backend crnn
```

Estimation accepts `ff` or `cnn`; tracking accepts `rnn` or `crnn`. Passing a backend that
does not match the task is rejected rather than silently ignored.


## Training

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
the default; `legacy` is the larger variant the first PyTorch port drifted into, whose
first dense layer alone holds 53.5M parameters. Checkpoints are routed to the right
topology automatically, so either loads without a flag.


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


