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


# The phone classes used for the published per-class tracking tables, taken
# verbatim from Tracker/LPC_model_apply.py.
PHONE_CLASSES = {
    "vowels": ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'er', 'ey', 'ih',
               'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux', 'axh'],
    "semivowels": ["l", "r", "er", "w", "y", "el", "hh", "hv"],
    "nasal": ["em", "en", "eng", "m", "n", "ng", "nx"],
    "fricatives": ["s", "z", "sh", "zh", "f", "th", "v", "dh", "em", "en", "eng",
                   "m", "n", "ng", "nx"],
    "affricates": ["ch", "jh"],
    "stops": ['b', "d", "g", "p", "t", "k", "dx", "q"],
}


def mae_hz(pred, target):
    mask = target > 0
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan
    return 1000.0 * (np.abs(pred - target) * mask).sum(axis=0) / counts


def phones_for(stem, label_root):
    """Per-frame phone labels for an utterance named dr<N>_<spk>_<utt>."""
    dr, spk, utt = stem.split('_')[:3]
    path = os.path.join(label_root, dr, spk, utt + '.label')
    if not os.path.exists(path):
        return None
    out = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts and parts[0]:
                out.append(parts[0])
    return out


def report_by_phone(phones, pred, target):
    """Per-class MAE in Hz, in the style of the published tracking table."""
    phones = np.asarray(phones)
    print(f"{'class':<12} {'n':>7}  {'F1':>6} {'F2':>6} {'F3':>6} {'F4':>6}")
    for name, members in PHONE_CLASSES.items():
        sel = np.isin(phones, members)
        if not sel.any():
            continue
        e = mae_hz(pred[sel], target[sel])
        print(f"{name:<12} {int(sel.sum()):>7}  "
              f"{e[0]:>6.1f} {e[1]:>6.1f} {e[2]:>6.1f} {e[3]:>6.1f}")


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


def eval_rnn(data, model_path, device, seq_len=20, label_root=None):
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

    if label_root:
        # Row names look like dr2_mwew0_sx191_frame_7.
        cache, phones = {}, []
        for name in rows[:n, 0]:
            stem, _, idx = str(name).rpartition('_frame_')
            if stem not in cache:
                cache[stem] = phones_for(stem, label_root) or []
            seq = cache[stem]
            phones.append(seq[int(idx)] if idx.isdigit() and int(idx) < len(seq) else '')
        report_by_phone(phones, P, Y[:n])
    return mae_hz(P, Y[:n]), n


def eval_crnn(data, model_path, device, label_root=None):
    from crnn_inference import load_model, predict_patches
    entries = np.load(data, allow_pickle=True)
    model = load_model(model_path, device)
    all_pred, all_lab, all_phones = [], [], []
    for name, labels, patches in entries:
        n = min(len(patches), len(labels))
        if n == 0:
            continue
        all_pred.append(predict_patches(model, patches[:n], device)[:n])
        all_lab.append(labels[:n])
        if label_root:
            seq = phones_for(str(name), label_root) or []
            all_phones.extend(seq[:n] + [''] * max(0, n - len(seq)))
    P = np.concatenate(all_pred)
    Y = np.concatenate(all_lab)
    if label_root:
        report_by_phone(all_phones, P, Y)
    return mae_hz(P, Y), len(Y)


EVALS = {'ff': eval_ff, 'cnn': eval_cnn, 'rnn': eval_rnn, 'crnn': eval_crnn}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a formant model.")
    parser.add_argument('--backend', required=True, choices=sorted(EVALS))
    parser.add_argument('--data', required=True, help="prepared test split for this backend")
    parser.add_argument('--model', default=None, help="checkpoint (default: resolved)")
    parser.add_argument('--label_root', default=None,
                        help="VTR Test/ directory; adds the per-phone-class breakdown "
                             "used by the published tracking table (rnn/crnn only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {}
    if args.label_root:
        if args.backend not in ('rnn', 'crnn'):
            parser.error("--label_root only applies to the tracking backends (rnn, crnn).")
        kwargs['label_root'] = args.label_root
    errors, n = EVALS[args.backend](args.data, args.model, device, **kwargs)
    print(f"{args.backend:5s} n={n:<7d} MAE Hz: "
          f"F1 {errors[0]:6.1f}  F2 {errors[1]:6.1f}  "
          f"F3 {errors[2]:6.1f}  F4 {errors[3]:6.1f}  | mean {np.nanmean(errors):.1f}")


if __name__ == "__main__":
    main()
