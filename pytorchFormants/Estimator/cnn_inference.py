"""CNN formant estimation inference, straight from a WAV file.

The previous spectrogramEstimate_inference.py could only score a prebuilt .npy,
so the CNN estimator was unreachable from formants.py. This takes a wav plus an
optional time window, builds the 55x50 spectrogram image and predicts F1-F4.

Usage:
    python cnn_inference.py <wav_file> <predictions_file> [--begin B --end E] [--model M]
"""
from __future__ import absolute_import, print_function

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_model import build_model, infer_arch
from spec_features import image_from_wav

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "CNN_estimate.pt")


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train one with cnn_train.py, "
            "or fetch the published weights (see README).")
    state = torch.load(model_path, map_location=device)
    model = build_model(infer_arch(state)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_from_wav(wav_filename, preds_file, begin=None, end=None,
                     model_path=DEFAULT_MODEL):
    image = image_from_wav(wav_filename, begin, end)
    if image is None:
        raise ValueError(f"{wav_filename} segment is shorter than one 10ms frame.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    x = torch.from_numpy(image[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    with open(preds_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['NAME', 'F1', 'F2', 'F3', 'F4'])
        writer.writerow([wav_filename] + (1000 * pred).tolist())
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN formant estimation from a WAV file.")
    parser.add_argument('wav_file')
    parser.add_argument('predictions_file')
    parser.add_argument('--begin', type=float, default=None)
    parser.add_argument('--end', type=float, default=None)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()

    predict_from_wav(args.wav_file, args.predictions_file,
                     args.begin, args.end, args.model)
