"""Train the CRNN formant tracker from scratch.

Build the inputs first with crnn_feats.py, then:

    python crnn_train.py --train_data data/crnn/Train_crnn.npy \
                         --test_data  data/crnn/Test_crnn.npy

The original Keras model was trained with mean absolute error and adagrad; the
same is used here. Labels of -1 mark unannotated formants (common in Hillenbrand
and Clopper, absent in VTR) and are masked out of the loss rather than being
treated as a target of -1 kHz.
"""
from __future__ import absolute_import, print_function

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crnn_feats import N_FREQ, N_TIME, SEQ_LEN
from crnn_model import CRNN


def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found at {path}. Run crnn_feats.py first.")
    return np.load(path, allow_pickle=True)


def make_sequences(entries, seq_len):
    """Cut utterances into fixed-length sequences of patches.

    Returns X (N, seq_len, 1, N_FREQ, N_TIME) and Y (N, seq_len, 4).
    """
    xs, ys = [], []
    for _, labels, patches in entries:
        n = min(len(patches), len(labels)) // seq_len
        for c in range(n):
            lo, hi = c * seq_len, (c + 1) * seq_len
            xs.append(patches[lo:hi])
            ys.append(labels[lo:hi])
    if not xs:
        raise ValueError(f"No utterance was long enough for seq_len={seq_len}.")
    X = np.asarray(xs, dtype=np.float32)[:, :, None, :, :]
    return X, np.asarray(ys, dtype=np.float32)


def masked_l1(pred, target):
    """L1 over annotated entries only; -1 marks 'not annotated'."""
    mask = (target > 0).float()
    denom = mask.sum()
    if denom == 0:
        return pred.sum() * 0.0
    return ((pred - target).abs() * mask).sum() / denom


def evaluate(model, entries, device, seq_len, batch_sequences=64):
    """Per-formant MAE in Hz over whole utterances, ignoring unannotated frames."""
    from crnn_inference import predict_patches

    model.eval()
    err = np.zeros(4, dtype=np.float64)
    cnt = np.zeros(4, dtype=np.float64)
    with torch.no_grad():
        for _, labels, patches in entries:
            n = min(len(patches), len(labels))
            if n == 0:
                continue
            pred = predict_patches(model, patches[:n], device, batch_sequences)[:n]
            lab = labels[:n]
            m = lab > 0
            err += (np.abs(pred - lab) * m).sum(axis=0)
            cnt += m.sum(axis=0)
    model.train()
    cnt[cnt == 0] = np.nan
    return 1000.0 * err / cnt


def main():
    parser = argparse.ArgumentParser(description="Train the CRNN formant tracker.")
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output', default="CRNN_tracker_retrained.pt")
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_len', default=SEQ_LEN, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    if os.path.abspath(args.train_data) == os.path.abspath(args.test_data):
        parser.error("--train_data and --test_data point at the same file; "
                     "test scores would be measured on training data.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_entries = load_split(args.train_data)
    test_entries = load_split(args.test_data)
    Xtrain, Ytrain = make_sequences(train_entries, args.seq_len)
    print(f"train sequences: {Xtrain.shape}, test utterances: {len(test_entries)}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="asr-formants")
        run.config.update({"lr": args.lr, "batch_size": args.batch_size,
                           "epochs": args.epochs, "seq_len": args.seq_len, "model": "CRNN"})

    model = CRNN(n_freq=N_FREQ, n_time=N_TIME).to(device)
    if run is not None:
        run.watch(model)
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    n_batches = int(np.ceil(len(Xtrain) / args.batch_size))
    best = float('inf')

    print("Starting training")
    for epoch in range(args.epochs):
        order = np.random.permutation(len(Xtrain))
        cost = 0.0
        for b in range(n_batches):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            xb = torch.from_numpy(Xtrain[idx]).to(device)
            yb = torch.from_numpy(Ytrain[idx]).to(device)

            optimizer.zero_grad()
            loss = masked_l1(model(xb), yb)
            loss.backward()
            optimizer.step()
            cost += loss.item()

        errors = evaluate(model, test_entries, device, args.seq_len)
        score = float(np.nanmean(errors))
        print(f"Epoch {epoch}: train loss {cost / n_batches:.4f} | "
              f"test MAE Hz F1 {errors[0]:.1f} F2 {errors[1]:.1f} "
              f"F3 {errors[2]:.1f} F4 {errors[3]:.1f} | mean {score:.1f}")

        if run is not None:
            run.log({"train_loss": cost / n_batches, "mean_abs_error_hz": score,
                     "F1": errors[0], "F2": errors[1], "F3": errors[2], "F4": errors[3]})

        # Only overwrite the checkpoint when the model actually improved.
        if score < best:
            best = score
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (mean test MAE {best:.1f} Hz)")

    print(f"Best mean test MAE: {best:.1f} Hz -> {args.output}")


if __name__ == "__main__":
    main()
