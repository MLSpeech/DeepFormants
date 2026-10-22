"""CRNN formant tracker: a faithful PyTorch port of the original Keras model.

The original lives in the project archive as old/Tracker/SingleCNNTracker.py with
freq_norm_cnn_model2.json / freq_norm_cnn_weights_best2.h5 (Keras 1.2.2, Theano
dim ordering). Architecture:

    input (batch, 10, 1, 55, 50)
    ConvLSTM2D(64, 3x3, border_mode='same', activation='relu',
               inner_activation='hard_sigmoid', return_sequences=True)
    TimeDistributed[ Conv2D(32, 3x3, valid) -> MaxPool(2,2) ] x4   # 55x50 -> 1x1
    TimeDistributed[ Flatten -> Dropout(.05) -> Dense(256, relu)
                     -> Dropout(.05) -> Dense(4, linear) ]

The gate semantics below reproduce Keras 1.2.2 exactly so the published weights
can be loaded unchanged; see convert_keras_weights.py.
"""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FREQ = 55
N_TIME = 50
SEQ_LEN = 10


def hard_sigmoid(x):
    """Keras 1.x hard_sigmoid: clip(0.2*x + 0.5, 0, 1)."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class ConvLSTM2DCell(nn.Module):
    """Single ConvLSTM2D step with Keras 1.2.2 gate ordering and activations.

    Keras computes, per timestep:
        i = inner_act(conv(x, W_i) + b_i + conv(h, U_i))
        f = inner_act(conv(x, W_f) + b_f + conv(h, U_f))
        c = f * c_prev + i * act(conv(x, W_c) + b_c + conv(h, U_c))
        o = inner_act(conv(x, W_o) + b_o + conv(h, U_o))
        h = o * act(c)
    with act=relu and inner_act=hard_sigmoid here. Note the recurrent
    convolutions carry no bias.
    """

    def __init__(self, in_channels=1, hidden_channels=64, kernel_size=3):
        super(ConvLSTM2DCell, self).__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2  # border_mode='same'

        # Input-to-hidden convolutions (with bias), one per gate.
        for g in ('i', 'f', 'c', 'o'):
            setattr(self, 'W_' + g,
                    nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad, bias=True))
        # Hidden-to-hidden convolutions (no bias), one per gate.
        for g in ('i', 'f', 'c', 'o'):
            setattr(self, 'U_' + g,
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad, bias=False))

    def forward(self, x, state):
        h_prev, c_prev = state
        i = hard_sigmoid(self.W_i(x) + self.U_i(h_prev))
        f = hard_sigmoid(self.W_f(x) + self.U_f(h_prev))
        o = hard_sigmoid(self.W_o(x) + self.U_o(h_prev))
        c = f * c_prev + i * F.relu(self.W_c(x) + self.U_c(h_prev))
        h = o * F.relu(c)
        return h, c


class CRNN(nn.Module):

    def __init__(self, n_freq=N_FREQ, n_time=N_TIME, hidden_channels=64, output_dim=4,
                 dropout=0.05):
        super(CRNN, self).__init__()
        self.n_freq = n_freq
        self.n_time = n_time

        self.convlstm = ConvLSTM2DCell(1, hidden_channels, 3)

        self.conv1 = nn.Conv2d(hidden_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.flat_dim = self._infer_flat_dim(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def _infer_flat_dim(self, hidden_channels):
        with torch.no_grad():
            d = torch.zeros(1, hidden_channels, self.n_freq, self.n_time)
            return self._conv_stack(d).shape[1]

    def _conv_stack(self, x):
        """Per-frame conv tower; 55x50 collapses to 1x1 with 32 channels."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        return x.reshape(x.size(0), -1)

    def forward(self, x):
        """x: (batch, seq, 1, n_freq, n_time) -> (batch, seq, 4)"""
        batch, seq = x.shape[0], x.shape[1]
        h = x.new_zeros(batch, self.convlstm.hidden_channels, self.n_freq, self.n_time)
        c = torch.zeros_like(h)

        hs = []
        for t in range(seq):
            h, c = self.convlstm(x[:, t], (h, c))
            hs.append(h)
        # (batch*seq, channels, freq, time) so the tower runs on every frame at once.
        hs = torch.stack(hs, dim=1).reshape(batch * seq, *hs[0].shape[1:])

        y = self._conv_stack(hs)
        y = self.dropout(y)
        y = F.relu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        return y.reshape(batch, seq, -1)
