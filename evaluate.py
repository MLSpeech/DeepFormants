"""Evaluate any of the four models on a prepared VTR test split.

    python evaluate.py --backend ff   --data .../Outputs/Test.npy
    python evaluate.py --backend cnn  --data data/cnn/vtr_test_X.npy
    python evaluate.py --backend rnn  --data .../Outputs/Test_tracker.npy
    python evaluate.py --backend crnn --data data/crnn/Test_crnn.npy

Reports per-formant mean absolute error in Hz. Frames where a formant is not
annotated (-1) are excluded. For reference, the published VTR estimation errors
are F1 50 / F2 86 / F3 104 Hz, and the tracking errors are reported per phone
class (vowels F1 54 / F2 81 / F3 112).
"""
from __future__ import absolute_import, print_function

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "pytorchFormants"))
sys.path.append(os.path.join(HERE, "pytorchFormants", "Estimator"))
sys.path.append(os.path.join(HERE, "pytorchFormants", "Tracker"))
sys.path.append(os.path.join(HERE, "pytorchFormants", "CRNN"))

from weights import resolve


def mae_hz(pred, target):
    mask = target > 0
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan
    return 1000.0 * (np.abs(pred - target) * mask).sum(axis=0) / counts


def eval_ff(data, model_path, device):
    from ff_model import FFEstimator
    rows = np.load(data, allow_pickle=True)
    X = np.nan_to_num(rows[:, 5:].astype(np.float32))
    Y = rows[:, 1:5].astype(np.float32)
    model = FFEstimator(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load(resolve('ff', model_path), map_location=device))
    model.eval()
    with torch.no_grad():
        P = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return mae_hz(P, Y), len(Y)


def eval_cnn(data, model_path, device):
    from cnn_model import build_model, infer_arch
    X = np.load(data).astype(np.float32)
    Y_raw = np.load(data.replace('_X.npy', '_Y.npy'), allow_pickle=True)
    Y = np.array([[float(v) for v in r[1:5]] for r in Y_raw], dtype=np.float32)
    state = torch.load(resolve('cnn', model_path), map_location=device)
    model = build_model(infer_arch(state)).to(device)
    model.load_state_dict(state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 500):
            preds.append(model(torch.from_numpy(X[i:i + 500]).to(device)).cpu().numpy())
    return mae_hz(np.concatenate(preds), Y), len(Y)


def eval_rnn(data, model_path, device, seq_len=20):
    from rnn_model import LSTMTracker
    rows = np.load(data, allow_pickle=True)
    X = np.nan_to_num(rows[:, 5:].astype(np.float32))
    Y = rows[:, 1:5].astype(np.float32)
    n = (len(X) // seq_len) * seq_len
    Xs = X[:n].reshape(-1, seq_len, X.shape[1])
    model = LSTMTracker(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load(resolve('rnn', model_path), map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 500):
            preds.append(model(torch.from_numpy(Xs[i:i + 500]).to(device)).cpu().numpy())
    P = np.concatenate(preds).reshape(-1, 4)
    return mae_hz(P, Y[:n]), n


def eval_crnn(data, model_path, device):
    from crnn_inference import load_model, predict_patches
    entries = np.load(data, allow_pickle=True)
    model = load_model(model_path, device)
    err = np.zeros(4)
    cnt = np.zeros(4)
    total = 0
    for _, labels, patches in entries:
        n = min(len(patches), len(labels))
        if n == 0:
            continue
        P = predict_patches(model, patches[:n], device)[:n]
        lab = labels[:n]
        m = lab > 0
        err += (np.abs(P - lab) * m).sum(axis=0)
        cnt += m.sum(axis=0)
        total += n
    cnt[cnt == 0] = np.nan
    return 1000.0 * err / cnt, total


EVALS = {'ff': eval_ff, 'cnn': eval_cnn, 'rnn': eval_rnn, 'crnn': eval_crnn}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a formant model.")
    parser.add_argument('--backend', required=True, choices=sorted(EVALS))
    parser.add_argument('--data', required=True, help="prepared test split for this backend")
    parser.add_argument('--model', default=None, help="checkpoint (default: resolved)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors, n = EVALS[args.backend](args.data, args.model, device)
    print(f"{args.backend:5s} n={n:<7d} MAE Hz: "
          f"F1 {errors[0]:6.1f}  F2 {errors[1]:6.1f}  "
          f"F3 {errors[2]:6.1f}  F4 {errors[3]:6.1f}  | mean {np.nanmean(errors):.1f}")


if __name__ == "__main__":
    main()
