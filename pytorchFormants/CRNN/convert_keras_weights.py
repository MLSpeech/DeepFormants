"""Convert the original Keras CRNN weights (freq_norm_cnn_weights_best2.h5) to PyTorch.

The source model is Keras 1.2.2 with Theano dim ordering, which stores conv
kernels as (out_channels, in_channels, rows, cols) -- the same layout PyTorch
uses. However Theano performs a true convolution (filter_flip=True) whereas
PyTorch's conv2d is a cross-correlation, so every conv kernel must be flipped
on both spatial axes during conversion. Dense weights are (in, out) in Keras
and (out, in) in PyTorch, so those are transposed.

Verified: with the flip, this port reproduces the original model's committed
predictions (old/Tracker/Predictions*.csv) to ~0.002 Hz.

Usage:
    python convert_keras_weights.py <freq_norm_cnn_weights_best2.h5> <out.pt>
"""
from __future__ import absolute_import, print_function

import os
import sys

import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crnn_model import CRNN

# Keras layer/weight name -> PyTorch parameter name.
CONV_MAP = {
    'timedistributed_1/convolution2d_1': 'conv1',
    'timedistributed_3/convolution2d_2': 'conv2',
    'timedistributed_5/convolution2d_3': 'conv3',
    'timedistributed_7/convolution2d_4': 'conv4',
}
DENSE_MAP = {
    'timedistributed_11/dense_1': 'fc1',
    'timedistributed_13/dense_2': 'fc2',
}


def _conv_kernel(dataset):
    """Theano conv kernel -> PyTorch: flip both spatial axes."""
    return torch.from_numpy(np.array(dataset)[:, :, ::-1, ::-1].copy())


def convert(h5_path, out_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Keras weights not found at {h5_path}")

    model = CRNN()
    sd = model.state_dict()

    with h5py.File(h5_path, 'r') as f:
        w = f['model_weights']

        # ConvLSTM2D: W_* are input kernels (with bias), U_* are recurrent (no bias).
        base = 'convlstm2d_1/convlstm2d_1_'
        for gate in ('i', 'f', 'c', 'o'):
            sd[f'convlstm.W_{gate}.weight'] = _conv_kernel(w[base + 'W_' + gate])
            sd[f'convlstm.W_{gate}.bias'] = torch.from_numpy(np.array(w[base + 'b_' + gate]))
            sd[f'convlstm.U_{gate}.weight'] = _conv_kernel(w[base + 'U_' + gate])

        for k_name, t_name in CONV_MAP.items():
            sd[f'{t_name}.weight'] = _conv_kernel(w[f'{k_name}_W'])
            sd[f'{t_name}.bias'] = torch.from_numpy(np.array(w[f'{k_name}_b']))

        for k_name, t_name in DENSE_MAP.items():
            # Keras Dense is (in, out); PyTorch Linear is (out, in).
            sd[f'{t_name}.weight'] = torch.from_numpy(np.array(w[f'{k_name}_W']).T.copy())
            sd[f'{t_name}.bias'] = torch.from_numpy(np.array(w[f'{k_name}_b']))

    model.load_state_dict(sd)
    torch.save(model.state_dict(), out_path)
    print(f"Wrote {out_path}")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_keras_weights.py <keras_weights.h5> <out.pt>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
