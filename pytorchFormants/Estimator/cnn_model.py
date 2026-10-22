"""CNN formant estimator architectures.

Two variants exist for historical reasons:

  'keras'  - matches the original Keras estimator (old/Estimator/cnn_normalized.py
             and cnn_freq_norm_stableY.py): Conv(96,'same') -> Conv(32) -> Pool(2,2)
             -> Conv(64,'same') -> Conv(64) -> Pool(2,2) -> Dense(512) -> Dense(4).
  'legacy' - the shape the PyTorch port drifted into (spectrogramEstimate.py):
             all-valid convolutions, stride-1 2x2 pooling and a 5x5 fourth conv.
             The committed CNN_estimate.pt uses this variant.

The legacy flatten is 43*38*64 = 104576, so its first dense layer alone holds
53.5M parameters and the checkpoint is 215MB. The keras variant flattens to
10*11*64 = 7040 and is roughly 20x smaller for equivalent accuracy.
"""
from __future__ import absolute_import, print_function

import torch.nn as nn
import torch.nn.functional as F


class CNNKeras(nn.Module):
    """Original Keras estimator topology.

    For the standard 55x50 input the conv stack flattens to 64*12*11 = 8448.
    """

    def __init__(self, output_dim=4, n_freq=55, n_time=50):
        super(CNNKeras, self).__init__()
        self.Conv1 = nn.Conv2d(1, 96, 3, padding=1)   # border_mode='same'
        self.Conv2 = nn.Conv2d(96, 32, 3)
        self.Conv3 = nn.Conv2d(32, 64, 3, padding=1)  # border_mode='same'
        self.Conv4 = nn.Conv2d(64, 64, 3)
        self.Dense5 = nn.Linear(self._flat_dim(n_freq, n_time), 512)
        self.out = nn.Linear(512, output_dim)

    def _flat_dim(self, n_freq, n_time):
        import torch
        with torch.no_grad():
            return self._features(torch.zeros(1, 1, n_freq, n_time)).shape[1]

    def _features(self, x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.max_pool2d(x, 2)
        return x.reshape(x.size(0), -1)

    def forward(self, x):
        x = F.relu(self.Dense5(self._features(x)))
        return self.out(x)


class CNNLegacy(nn.Module):
    """The variant the committed CNN_estimate.pt was trained with."""

    def __init__(self, output_dim=4):
        super(CNNLegacy, self).__init__()
        self.Conv1 = nn.Conv2d(1, 96, (3, 3))
        self.Conv2 = nn.Conv2d(96, 32, (3, 3))
        self.Conv3 = nn.Conv2d(32, 64, (3, 3))
        self.Conv4 = nn.Conv2d(64, 64, (5, 5))
        self.Dense5 = nn.Linear(43 * 38 * 64, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.relu(self.Conv3(x))
        x = F.relu(self.Conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.Dense5(x))
        return self.out(x)


ARCHS = {'keras': CNNKeras, 'legacy': CNNLegacy}


def build_model(arch='keras', output_dim=4):
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; expected one of {sorted(ARCHS)}")
    return ARCHS[arch](output_dim=output_dim)


def infer_arch(state_dict):
    """Pick the architecture that matches a checkpoint's Dense5 shape.

    The two variants are told apart by their flatten width: legacy is
    43*38*64 = 104576, keras is 12*11*64 = 8448 for a 55x50 input.
    """
    dense = state_dict.get('Dense5.weight')
    if dense is None:
        raise ValueError("Checkpoint has no Dense5.weight; is this a CNN estimator?")
    return 'legacy' if dense.shape[1] == 43 * 38 * 64 else 'keras'
