"""LSTM formant tracker: two recurrent layers over the 350-dim LPC/cepstral vector."""
from __future__ import absolute_import, print_function

import torch.nn as nn


class LSTMTracker(nn.Module):

    def __init__(self, input_dim=350, output_dim=4):
        super(LSTMTracker, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x)
