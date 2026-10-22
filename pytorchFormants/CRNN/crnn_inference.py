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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crnn_feats import SEQ_LEN, group_sequences, patches_from_wav
from crnn_model import CRNN
from weights import resolve

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = None  # resolved via pytorchFormants.weights


def load_model(model_path, device):
    model_path = resolve('crnn', model_path)
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_patches(model, patches, device, batch_sequences=64):
    """Run the model over one utterance's patches, returning (n_frames, 4) in kHz."""
    seqs, n_real = group_sequences(patches, SEQ_LEN)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_sequences):
            chunk = torch.from_numpy(seqs[i:i + batch_sequences]).to(device)
            out.append(model(chunk).reshape(-1, 4).cpu().numpy())
    return np.concatenate(out)[:n_real]


def predict_from_wav(wav_filename, preds_file, model_path=DEFAULT_MODEL):
    if not os.path.exists(wav_filename):
        raise FileNotFoundError(f"WAV file not found at {wav_filename}")

    patches, _ = patches_from_wav(wav_filename)
    if len(patches) == 0:
        raise ValueError(f"{wav_filename} is too short to produce any frame.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    preds = predict_patches(model, patches, device)

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
