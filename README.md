DeepFormants
============

Shua Dissen (shua.dissen@gmail.com)            
Joseph Keshet (joseph.keshet@biu.ac.il)  


DeepFormants is a software package for formant tracking and estimation, using two algorithms based on deep networks. It works as follows:
* The user provides a wav file with an initial stop consonant. 
* Optionally, the user can designate a window for estimation by providing a start and end time (specified in seconds).
* A classifer is used to estimate the formants in the file, with two modes:
* Estimation: If a time window is specified, a single estimate is made for F1-F4 within that window. 
* Tracking: If no time window is given, the model will track F1-F4 and give a measurement at every 10 milliseconds across the length of the file.

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

The estimation and tracking weights are committed under `pytorchFormants/`
(`Estimator/LPC_NN_scaledLoss.pt` and `Tracker/LPC_RNN.pt`) and are loaded automatically.
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

This uses the LSTM tracker over LPC features (`--backend lpc`, the default).


## CRNN tracker

An alternative tracker is available: a convolutional frontend over the spectrogram feeding
the recurrent head. The conv channel progression is the one from the CNN estimator
(`spectrogramEstimate.py`) and the recurrent head is the one from the LSTM tracker
(`LPC_tracker.py`); pooling is applied on the frequency axis only, so the model still emits
one F1-F4 prediction per 10 ms frame. Its frame grid matches the LPC tracker's exactly.

No CRNN weights are committed, so train one first and pass it with `--model`.

1. Extract spectrogram features. Expects a directory tree of WAVs each with a matching
   `.label` file (tab separated: phone, F1, F2, F3, F4 in Hz, one row per 10 ms):

```
python pytorchFormants/CRNN/crnn_feats.py --train_dir /path/Train/ --test_dir /path/Test/ \
          --output_dir /path/Outputs/
```

2. Train:

```
python pytorchFormants/CRNN/crnn_train.py --train_data /path/Outputs/Train_crnn.npy \
          --test_data /path/Outputs/Test_crnn.npy --output CRNN_tracker.pt
```

Test error is reported per formant in Hz, measured over whole utterances. Add `--wandb` to
log to Weights & Biases.

3. Track with it, either through the main script:

```
python formants.py data/Example.wav data/ExamplePredictions.csv --backend crnn \
          --model CRNN_tracker.pt
```

or directly:

```
python pytorchFormants/CRNN/crnn_inference.py data/Example.wav preds.csv CRNN_tracker.pt
```


## TODO

Training code for the estimator and the LSTM tracker lives under `pytorchFormants/` but
still reads hard-coded dataset paths; it needs the same `--train_data/--test_data`
treatment as `crnn_train.py`.


