"""Spectrogram feature extraction for the CRNN formant tracker.

The CRNN consumes a log-magnitude spectrogram instead of the 350-dimensional
LPC/cepstral vector used by the feed-forward estimator and the LSTM tracker.
The conv frontend needs the raw frequency axis to convolve over, so the
cepstral compression done in extract_features.py is deliberately skipped here.

Framing is kept identical to extract_features.build_data (30 ms window, 10 ms
hop) so that frame i of the CRNN lines up with frame i of the LPC tracker and
with row i of a VTR .label file.
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
WINDOW = 480          # 30 ms, matches extract_features.build_data
HOP = 160             # 10 ms, one frame per VTR label row
NFFT = 512
N_BINS = 160          # 160 * 16000/512 = 0-5000 Hz, enough headroom for F1-F4
EPSILON = 1e-10


def read_wav(wav_filename):
    """Read a 16 kHz mono WAV into a float array."""
    with wave.open(wav_filename, 'rb') as wav_in_file:
        n_samples = wav_in_file.getnframes()
        dstr = wav_in_file.readframes(n_samples)
    return np.frombuffer(dstr, np.int16).astype(np.float64)


def to_16k_mono(input_wav_filename):
    """Convert to 16 kHz mono via sox, as the rest of the package does.

    Returns (path, is_temporary).
    """
    tmp_wav16_filename = generate_tmp_filename("wav")
    easy_call("sox " + input_wav_filename + " -c 1 -r " + str(SAMPLE_RATE) + " " + tmp_wav16_filename)
    return tmp_wav16_filename


def num_frames(n_samples):
    """Frame count used by extract_features.build_data."""
    return len(range(0, n_samples - 100, HOP))


def spectrogram(data, normalize=True):
    """Log-magnitude spectrogram of a waveform.

    Returns an (N_BINS, T) float32 array, where T matches num_frames(len(data)).
    """
    window = np.hamming(WINDOW)
    frames = []
    for start in range(0, len(data) - 100, HOP):
        frame = data[start:start + WINDOW]
        if len(frame) < WINDOW:
            # The tail frame can be short; zero-pad so every frame has the
            # same length and the frame count still matches build_data.
            frame = np.pad(frame, (0, WINDOW - len(frame)))
        spec = np.abs(np.fft.rfft(frame * window, NFFT)[:N_BINS]) ** 2
        frames.append(np.log(spec + EPSILON))

    if not frames:
        return np.zeros((N_BINS, 0), dtype=np.float32)

    spec = np.asarray(frames, dtype=np.float32).T  # (N_BINS, T)
    if normalize:
        # Per-utterance mean/variance normalization, so the conv frontend sees
        # a comparable dynamic range at train and inference time.
        spec = (spec - spec.mean()) / (spec.std() + EPSILON)
    return np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)


def spectrogram_from_wav(input_wav_filename, normalize=True, convert=True):
    """Spectrogram for a WAV file, resampling to 16 kHz mono first."""
    tmp_wav16_filename = None
    try:
        if convert:
            tmp_wav16_filename = to_16k_mono(input_wav_filename)
            path = tmp_wav16_filename
        else:
            path = input_wav_filename
        return spectrogram(read_wav(path), normalize=normalize)
    finally:
        if tmp_wav16_filename and os.path.exists(tmp_wav16_filename):
            os.remove(tmp_wav16_filename)


def load_labels(label_file):
    """Per-frame F1-F4 in kHz, one row per 10 ms, as in Tracker/create_VTR_feats.py."""
    formants_list = []
    with open(label_file, 'r') as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            formants_list.append([float(v) / 1000.0 for v in row[1:5]])
    return np.asarray(formants_list, dtype=np.float32)


def process_directory(input_dir, output_file):
    """Build (name, labels, spectrogram) entries for every labelled WAV in a tree."""
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
            spec = spectrogram_from_wav(wav_file)

            # Frame count and label count can differ by a frame or two at the
            # tail; truncate to the shorter, as the other feature builders do.
            length = min(spec.shape[1], labels.shape[0])
            if length == 0:
                print(f"No usable frames in {wav_file}")
                continue

            name = '_'.join(wav_file.replace('.wav', '').split('/')[-3:])
            entries.append((name, labels[:length], spec[:, :length]))

    np.save(output_file, np.array(entries, dtype=object), allow_pickle=True)
    print(f"Saved {len(entries)} utterances to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract CRNN spectrogram features.")
    parser.add_argument('--train_dir', default="/home/datasets/public/formants/vtr/shua_processed/Train/",
                        help="directory tree of training WAVs with matching .label files")
    parser.add_argument('--test_dir', default="/home/datasets/public/formants/vtr/shua_processed/Test/",
                        help="directory tree of test WAVs with matching .label files")
    parser.add_argument('--output_dir', default="/home/datasets/public/formants/vtr/shua_processed/Outputs/",
                        help="where to write Train_crnn.npy / Test_crnn.npy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Processing Train directory...")
    process_directory(args.train_dir, os.path.join(args.output_dir, "Train_crnn.npy"))

    print("Processing Test directory...")
    process_directory(args.test_dir, os.path.join(args.output_dir, "Test_crnn.npy"))


if __name__ == "__main__":
    main()
