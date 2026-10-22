"""Spectrogram-image features for the CNN formant estimator.

Ported from the original old/Estimator/normalized_spectrogram.py so that the
published CNN weights keep working. One vowel segment becomes one 55x50 image:

  * pre-emphasis (0.97)
  * STFT with a 160-sample Hanning window and hop = len(segment)/50, so a
    segment of any duration yields ~50 frames
  * magnitude -> dB via 20*log10(|S| / 1e-5)
  * a single global mean subtracted from the whole image (note the CRNN
    *tracker* subtracts a per-time-frame mean instead; that difference is what
    "freq_norm" refers to in the tracker's filenames)
  * keep the first 55 bins (0-5.5 kHz), divide by 255
  * centre-pad or centre-crop to exactly 50 columns

This module intentionally has no training-time dependencies so inference can
import it on its own.
"""
from __future__ import absolute_import, print_function

import os
import wave

import numpy as np
from scipy.signal import lfilter

SAMPLE_RATE = 16000
FRAME_SIZE = 160
N_FREQ = 55
N_TIME = 50
DB_REF = 10e-6
PREEMPH = 0.97


def read_wav(wav_filename):
    with wave.open(wav_filename, 'rb') as f:
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, np.int16).astype(np.float64)


def preemp(x, p=PREEMPH):
    return lfilter([1.0, -p], 1, x)


def stft_fixed_frames(sig, frame_size=FRAME_SIZE, n_frames=N_TIME):
    """STFT whose hop is chosen so the signal yields about n_frames frames."""
    hop = max(1, int(len(sig) // n_frames))
    win = np.hanning(frame_size)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    cols = int(np.ceil((len(samples) - frame_size) / float(hop))) + 1
    samples = np.append(samples, np.zeros(frame_size))
    idx = np.arange(frame_size)[None, :] + hop * np.arange(cols)[:, None]
    return np.fft.rfft(samples[idx] * win)


def pad_to_width(image, width=N_TIME):
    """Centre an (N_FREQ, t) image inside an (N_FREQ, width) zero canvas."""
    canvas = np.zeros((image.shape[0], width), dtype=np.float32)
    if image.shape[1] > width:
        mid = image.shape[1] // 2
        image = image[:, mid - width // 2:mid + width // 2]
    pad = int(np.round((width - image.shape[1]) / 2.0))
    canvas[:, pad:pad + image.shape[1]] = image
    return canvas


def estimation_image(samples):
    """(N_FREQ, N_TIME) normalized spectrogram image for one vowel segment."""
    if len(samples) < FRAME_SIZE:
        return None
    spec = stft_fixed_frames(preemp(samples))
    with np.errstate(divide='ignore', invalid='ignore'):
        ims = 20.0 * np.log10(np.abs(spec) / DB_REF)
    ims = np.nan_to_num(ims, nan=0.0, posinf=0.0, neginf=0.0)
    ims = ims - ims.mean()           # single global mean, not per-frame
    ims = ims[:, :N_FREQ] / 255.0    # (frames, N_FREQ)
    return pad_to_width(ims.T.astype(np.float32))


def image_from_wav(wav_filename, begin=None, end=None):
    """Spectrogram image for a whole file, or for the [begin, end] second window."""
    if not os.path.exists(wav_filename):
        raise FileNotFoundError(f"WAV file not found at {wav_filename}")
    data = read_wav(wav_filename)
    if begin is not None and end is not None:
        data = data[int(begin * SAMPLE_RATE):int(end * SAMPLE_RATE)]
    return estimation_image(data)
