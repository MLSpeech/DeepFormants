"""Train the feed-forward formant estimator on LPC/cepstral features.

    python ff_train.py --train_data .../Outputs/Train.npy --test_data .../Outputs/Test.npy

Replaces LPC_Estimate_Class.py, which hard-coded its dataset paths and whose
"best model" guard only wrapped the print statement -- the checkpoint was
overwritten every epoch, so the saved weights were the last epoch rather than
the best one.

Input rows are [name, F1, F2, F3, F4, f_1 ... f_350] with formants in kHz, as
produced by Estimator/create_VTR_feats.py.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ff_model import FFEstimator
from splits import describe, split_rows, vtr_speaker


def load_raw(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run create_VTR_feats.py first.")
    data = np.load(path, allow_pickle=True)
    X = np.nan_to_num(data[:, 5:].astype(np.float32))
    Y = data[:, 1:5].astype(np.float32)
    return data[:, 0], X, Y


def load_split(path):
    _, X, Y = load_raw(path)
    return X, Y


def masked_l1(pred, target):
    mask = (target > 0).float()
    denom = mask.sum()
    if denom == 0:
        return pred.sum() * 0.0
    return ((pred - target).abs() * mask).sum() / denom


def evaluate(model, X, Y, device, batch=2000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            preds.append(model(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    model.train()
    P = np.concatenate(preds)
    mask = Y > 0
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan
    return 1000.0 * (np.abs(P - Y) * mask).sum(axis=0) / counts


def main():
    parser = argparse.ArgumentParser(description="Train the feed-forward formant estimator.")
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output', default="LPC_NN_retrained.pt")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_frac', default=0.15, type=float,
                        help="fraction of TRAINING speakers held out to choose the "
                             "checkpoint; the test split is scored once at the end")
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    if os.path.abspath(args.train_data) == os.path.abspath(args.test_data):
        parser.error("--train_data and --test_data point at the same file.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    names, Xall, Yall = load_raw(args.train_data)
    tr_mask, va_mask = split_rows(names, args.val_frac, args.seed)
    describe("ff", [vtr_speaker(n) for n in names[tr_mask]],
             [vtr_speaker(n) for n in names[va_mask]])

    Xtrain, Ytrain = Xall[tr_mask], Yall[tr_mask]
    Xval, Yval = Xall[va_mask], Yall[va_mask]
    Xtest, Ytest = load_split(args.test_data)
    print(f"train {Xtrain.shape}  val {Xval.shape}  test {Xtest.shape}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="asr-formants")
        run.config.update({"lr": args.lr, "batch_size": args.batch_size,
                           "epochs": args.epochs, "model": "FF"})

    model = FFEstimator(input_dim=Xtrain.shape[1]).to(device)
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

        # Selection uses the held-out validation speakers only.
        errors = evaluate(model, Xval, Yval, device)
        score = float(np.nanmean(errors))
        print(f"Epoch {epoch}: train loss {cost / n_batches:.4f} | val MAE Hz "
              f"F1 {errors[0]:.1f} F2 {errors[1]:.1f} F3 {errors[2]:.1f} F4 {errors[3]:.1f} "
              f"| mean {score:.1f}")

        if run is not None:
            run.log({"train_loss": cost / n_batches, "val_mean_abs_error_hz": score,
                     "val_F1": errors[0], "val_F2": errors[1],
                     "val_F3": errors[2], "val_F4": errors[3]})

        # Save only on improvement (the original saved unconditionally).
        if score < best:
            best = score
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (mean val MAE {best:.1f} Hz)")

    print(f"Best mean val MAE: {best:.1f} Hz -> {args.output}")

    # Test is scored exactly once, with the checkpoint chosen on validation.
    model.load_state_dict(torch.load(args.output, map_location=device))
    errors = evaluate(model, Xtest, Ytest, device)
    print(f"TEST (held out, scored once): F1 {errors[0]:.1f} F2 {errors[1]:.1f} "
          f"F3 {errors[2]:.1f} F4 {errors[3]:.1f} | mean {np.nanmean(errors):.1f}")
    if run is not None:
        run.log({"test_mean_abs_error_hz": float(np.nanmean(errors))})


if __name__ == "__main__":
    main()
