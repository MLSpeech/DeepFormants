"""Spectrogram features for the CRNN formant tracker.

A direct port of the feature pipeline in the original old/Tracker/SingleCNNTracker.py
so that the published weights produce the published numbers:

  * STFT with a 160-sample (10 ms) Hanning window and an 8-sample hop, zero
    padded by frameSize/2 at the front
  * magnitude -> dB via 20*log10(|S| / 1e-5)
  * per-time-frame mean subtraction (this is the "freq_norm" in the model name;
    note the CNN *estimator* subtracts a single global mean instead)
  * keep the first 55 bins (0-5.5 kHz at 100 Hz resolution), divide by 255
  * one 55x50 patch per output frame, centred on every 20th STFT frame
    (= every 10 ms) and zero padded to exactly 50 columns
  * patches grouped into sequences of 10 for the recurrent layer
"""
from __future__ import absolute_import, print_function

import argparse
import csv
import os
import sys
import wave

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helpers.utilities import easy_call, generate_tmp_filename

SAMPLE_RATE = 16000
FRAME_SIZE = 160        # 10 ms window
HOP = 8                 # STFT hop in samples
PATCH_STRIDE = 20       # 20 STFT frames * 8 samples = 160 samples = 10 ms
PATCH_HALF = 25         # +/- 25 STFT frames of context
N_FREQ = 55
N_TIME = 50
SEQ_LEN = 10
DB_REF = 10e-6


def read_wav(wav_filename):
    with wave.open(wav_filename, 'rb') as f:
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, np.int16).astype(np.float64)


def to_16k_mono(input_wav_filename):
    tmp = generate_tmp_filename("wav")
    easy_call("sox " + input_wav_filename + " -c 1 -r " + str(SAMPLE_RATE) + " " + tmp)
    return tmp


def stft(sig, frame_size=FRAME_SIZE, hop=HOP):
    """Short-time Fourier transform matching the original implementation."""
    win = np.hanning(frame_size)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    cols = int(np.ceil((len(samples) - frame_size) / float(hop))) + 1
    samples = np.append(samples, np.zeros(frame_size))
    idx = np.arange(frame_size)[None, :] + hop * np.arange(cols)[:, None]
    return np.fft.rfft(samples[idx] * win)


def spectrogram(data):
    """(T, N_FREQ) normalized dB spectrogram; T is the number of STFT frames."""
    spec = stft(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        ims = 20.0 * np.log10(np.abs(spec) / DB_REF)
    ims = np.nan_to_num(ims, nan=0.0, posinf=0.0, neginf=0.0)
    # Per-time-frame mean subtraction, then keep the low 55 bins.
    ims = ims - ims.mean(axis=1)[:, None]
    return ims[:, :N_FREQ] / 255.0


def patches_from_spectrogram(spec):
    """One (N_FREQ, N_TIME) patch per 10 ms frame, centred and zero padded."""
    patches, offsets = [], []
    for i in range(0, len(spec), PATCH_STRIDE):
        window = spec[max(0, i - PATCH_HALF):min(i + PATCH_HALF, len(spec))].T
        canvas = np.zeros((N_FREQ, N_TIME), dtype=np.float32)
        width = min(window.shape[1], N_TIME)
        if window.shape[1] > N_TIME:
            mid = window.shape[1] // 2
            window = window[:, mid - N_TIME // 2:mid + N_TIME // 2]
            width = window.shape[1]
        pad = int(np.round((N_TIME - width) / 2.0))
        canvas[:, pad:pad + width] = window[:, :width]
        patches.append(canvas)
        offsets.append(i)
    return np.asarray(patches, dtype=np.float32), offsets


def patches_from_wav(wav_filename, convert=True):
    tmp = None
    try:
        if convert:
            tmp = to_16k_mono(wav_filename)
            path = tmp
        else:
            path = wav_filename
        return patches_from_spectrogram(spectrogram(read_wav(path)))
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def group_sequences(patches, seq_len=SEQ_LEN):
    """Pad to a multiple of seq_len and reshape to (N, seq_len, 1, N_FREQ, N_TIME).

    Returns the grouped array and the number of real (non-padding) patches.
    """
    n_real = len(patches)
    pad = (-n_real) % seq_len
    if pad:
        patches = np.concatenate([patches, np.zeros((pad, N_FREQ, N_TIME), dtype=np.float32)])
    return patches.reshape(-1, seq_len, 1, N_FREQ, N_TIME), n_real


def load_labels(label_file):
    """Per-frame F1-F4 in kHz, one row per 10 ms. -1 marks an unannotated value."""
    out = []
    with open(label_file, 'r') as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            vals = [float(v) for v in row[1:5]]
            vals += [-1.0] * (4 - len(vals))
            out.append([v / 1000.0 if v > 0 else -1.0 for v in vals])
    return np.asarray(out, dtype=np.float32)


def process_directory(input_dir, output_file):
    """Build (name, labels, patches) entries for every labelled WAV in a tree."""
    entries = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if not file.endswith(".wav"):
                continue
            wav_file = os.path.join(root, file)
            label_file = os.path.splitext(wav_file)[0] + ".label"
            if not os.path.exists(label_file):
                print(f"Label file missing for {wav_file}")
                continue

            print(f"Processing: {wav_file}")
            labels = load_labels(label_file)
            patches, _ = patches_from_wav(wav_file)

            length = min(len(patches), len(labels))
            if length == 0:
                print(f"No usable frames in {wav_file}")
                continue

            name = '_'.join(wav_file.replace('.wav', '').split('/')[-3:])
            entries.append((name, labels[:length], patches[:length]))

    np.save(output_file, np.array(entries, dtype=object), allow_pickle=True)
    print(f"Saved {len(entries)} utterances to {output_file}")


def main():
    base = "/mlspeech/datasets/public/formants/vtr/shua_processed/"
    parser = argparse.ArgumentParser(description="Extract CRNN spectrogram patch features.")
    parser.add_argument('--train_dir', default=base + "Train/")
    parser.add_argument('--test_dir', default=base + "Test/")
    parser.add_argument('--output_dir', default="data/crnn/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Processing Train directory...")
    process_directory(args.train_dir, os.path.join(args.output_dir, "Train_crnn.npy"))
    print("Processing Test directory...")
    process_directory(args.test_dir, os.path.join(args.output_dir, "Test_crnn.npy"))


if __name__ == "__main__":
    main()
