"""Feed-forward formant estimator: dense net over the 350-dim LPC/cepstral vector."""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn


class FFEstimator(nn.Module):

    def __init__(self, input_dim=350, output_dim=4):
        super(FFEstimator, self).__init__()
        self.Dense1 = nn.Linear(input_dim, 1024)
        self.Dense2 = nn.Linear(1024, 512)
        self.Dense3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.Dense1(x))
        x = torch.sigmoid(self.Dense2(x))
        x = torch.sigmoid(self.Dense3(x))
        return self.out(x)
