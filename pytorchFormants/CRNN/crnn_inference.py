"""CRNN formant tracking inference.

Unlike the LPC tracker, this reads the waveform directly: the conv frontend
needs the spectrogram, not the cepstral feature CSV.

Usage:
    python crnn_inference.py <wav_file> <predictions_file> [model_path]
"""
from __future__ import absolute_import, print_function

import csv
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crnn_feats import spectrogram_from_wav
from crnn_model import CRNN

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "CRNN_tracker.pt")


def load_model(model_path, device, n_bins):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train one with crnn_train.py, "
            "or pass an explicit model path.")
    model = CRNN(n_bins=n_bins).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_from_wav(wav_filename, preds_file, model_path=DEFAULT_MODEL):
    if not os.path.exists(wav_filename):
        raise FileNotFoundError(f"WAV file not found at {wav_filename}")

    spec = spectrogram_from_wav(wav_filename)
    if spec.shape[1] == 0:
        raise ValueError(f"{wav_filename} is too short to produce any frame.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, spec.shape[0])

    # The whole utterance is fed as one sequence, so the recurrent layers see
    # full context rather than the fixed-length chunks used during training.
    inputs = torch.from_numpy(spec[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        preds = model(inputs).squeeze(0).cpu().numpy()

    with open(preds_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])
        for i, pred in enumerate(preds):
            writer.writerow([wav_filename + str(i)] + (1000 * pred).tolist())


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python crnn_inference.py <wav_file> <predictions_file> [model_path]")
        sys.exit(1)

    predict_from_wav(sys.argv[1], sys.argv[2],
                     sys.argv[3] if len(sys.argv) == 4 else DEFAULT_MODEL)
