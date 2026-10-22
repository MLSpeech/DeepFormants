"""Train the LSTM formant tracker on LPC/cepstral features.

    python rnn_train.py --train_data .../Outputs/Train_tracker.npy \
                        --test_data  .../Outputs/Test_tracker.npy

Replaces LPC_tracker.py, which loaded Train_tracker.npy as *both* splits -- so
every "test" score it printed was measured on training data, and the checkpoint
was selected on that. This script refuses to run when the two paths match.

Input rows are [name, F1, F2, F3, F4, f_1 ... f_350] with formants in kHz, as
produced by Tracker/create_VTR_feats.py.
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

from rnn_model import LSTMTracker


def load_split(path, seq_len):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run create_VTR_feats.py first.")
    data = np.load(path, allow_pickle=True)
    X = np.nan_to_num(data[:, 5:].astype(np.float32))
    Y = data[:, 1:5].astype(np.float32)
    n = (len(X) // seq_len) * seq_len
    return X[:n].reshape(-1, seq_len, X.shape[1]), Y[:n].reshape(-1, seq_len, 4)


def masked_l1(pred, target):
    mask = (target > 0).float()
    denom = mask.sum()
    if denom == 0:
        return pred.sum() * 0.0
    return ((pred - target).abs() * mask).sum() / denom


def evaluate(model, X, Y, device, batch=500):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            preds.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    model.train()
    P = np.concatenate(preds).reshape(-1, 4)
    Yf = Y.reshape(-1, 4)
    mask = Yf > 0
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan
    return 1000.0 * (np.abs(P - Yf) * mask).sum(axis=0) / counts


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM formant tracker.")
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output', default="LPC_RNN_retrained.pt")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--seq_len', default=20, type=int)
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

    Xtrain, Ytrain = load_split(args.train_data, args.seq_len)
    Xtest, Ytest = load_split(args.test_data, args.seq_len)
    print(f"train {Xtrain.shape}  test {Xtest.shape}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="asr-formants")
        run.config.update({"lr": args.lr, "batch_size": args.batch_size,
                           "epochs": args.epochs, "seq_len": args.seq_len, "model": "RNN"})

    model = LSTMTracker(input_dim=Xtrain.shape[2]).to(device)
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

        errors = evaluate(model, Xtest, Ytest, device)
        score = float(np.nanmean(errors))
        print(f"Epoch {epoch}: train loss {cost / n_batches:.4f} | test MAE Hz "
              f"F1 {errors[0]:.1f} F2 {errors[1]:.1f} F3 {errors[2]:.1f} F4 {errors[3]:.1f} "
              f"| mean {score:.1f}")

        if run is not None:
            run.log({"train_loss": cost / n_batches, "mean_abs_error_hz": score,
                     "F1": errors[0], "F2": errors[1], "F3": errors[2], "F4": errors[3]})

        if score < best:
            best = score
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (mean test MAE {best:.1f} Hz)")

    print(f"Best mean test MAE: {best:.1f} Hz -> {args.output}")


if __name__ == "__main__":
    main()
