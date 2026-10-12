"""CRNN formant tracker: convolutional frontend + recurrent head.

The two halves deliberately reuse the architectures already published in this
repo, so the CRNN is comparable with them rather than being a new design:

  * conv channel progression 1 -> 96 -> 32 -> 64 -> 64 is taken from the CNN
    estimator in spectrogramEstimate.py
  * the recurrent head (LSTM 512 -> LSTM 256 -> Linear 4) is taken from the
    LSTM tracker in pytorchFormants/Tracker/LPC_tracker.py

Pooling is applied on the frequency axis only. The time axis keeps full
resolution so the model still emits one F1-F4 prediction per 10 ms frame,
which is what the tracking task requires.
"""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINS = 160          # must match crnn_feats.N_BINS
N_POOLS = 4           # frequency axis is halved once per conv block
CONV_OUT_CHANNELS = 64


class CRNN(nn.Module):

    def __init__(self, n_bins=N_BINS, output_dim=4):
        super(CRNN, self).__init__()
        self.n_bins = n_bins

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 3), stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.Conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

        # After N_POOLS frequency halvings the conv stack emits
        # CONV_OUT_CHANNELS * (n_bins / 2**N_POOLS) features per frame.
        self.frame_dim = CONV_OUT_CHANNELS * (n_bins // (2 ** N_POOLS))

        self.lstm1 = nn.LSTM(input_size=self.frame_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        """x: (batch, 1, n_bins, time) -> (batch, time, 4)"""
        # Pool over frequency (dim 2) only, never over time (dim 3).
        x = F.max_pool2d(F.relu(self.Conv1(x)), kernel_size=(2, 1))
        x = F.max_pool2d(F.relu(self.Conv2(x)), kernel_size=(2, 1))
        x = F.max_pool2d(F.relu(self.Conv3(x)), kernel_size=(2, 1))
        x = F.max_pool2d(F.relu(self.Conv4(x)), kernel_size=(2, 1))

        # (batch, channels, freq, time) -> (batch, time, channels * freq)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x)
