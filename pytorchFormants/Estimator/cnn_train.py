"""Train the CNN formant estimator on spectrogram images.

Build the inputs first with build_cnn_data.py, then:

    python cnn_train.py --train_X data/cnn/merged_train_X.npy \
                        --train_Y data/cnn/merged_train_Y.npy \
                        --test_X  data/cnn/merged_test_X.npy  \
                        --test_Y  data/cnn/merged_test_Y.npy

Replaces spectrogramEstimate.py, which hard-coded its .npy filenames, saved a
checkpoint every epoch regardless of whether the model improved, and regressed
against -1 placeholders on corpora that do not annotate every formant.

--arch keras reproduces the original Keras estimator (same padding, 2x2 pooling,
3x3 final conv); --arch legacy keeps the all-valid / stride-1-pool variant that
the committed CNN_estimate.pt was trained with. The legacy variant's flatten is
43*38*64, which makes its first dense layer 53.5M parameters and the checkpoint
215MB; the keras variant is about 20x smaller.
"""
from __future__ import absolute_import, print_function

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_cnn_data import speaker_of
from cnn_model import build_model
from splits import describe, holdout_mask


def load_xy(x_path, y_path):
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Not found: {x_path}. Run build_cnn_data.py first.")
    X = np.load(x_path).astype(np.float32)
    Y_raw = np.load(y_path, allow_pickle=True)
    # Y rows are [name, F1, F2, F3, F4]; -1 marks an unannotated formant.
    Y = np.array([[float(v) for v in r[1:5]] for r in Y_raw], dtype=np.float32)
    names = np.array([str(r[0]) for r in Y_raw])
    return X, Y, names


def corpus_speaker(name):
    """Speaker key qualified by corpus, so ids cannot collide across corpora."""
    return f"{name.split('_')[0]}:{speaker_of(name)}"


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
    P = np.concatenate(preds)
    mask = Y > 0
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan
    return 1000.0 * (np.abs(P - Y) * mask).sum(axis=0) / counts


def main():
    parser = argparse.ArgumentParser(description="Train the CNN formant estimator.")
    parser.add_argument('--train_X', required=True)
    parser.add_argument('--train_Y', required=True)
    parser.add_argument('--test_X', required=True)
    parser.add_argument('--test_Y', required=True)
    parser.add_argument('--output', default="CNN_estimate_retrained.pt")
    parser.add_argument('--arch', default='keras', choices=['keras', 'legacy'])
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # Adagrad at the original 0.01 collapses on roughly half of seeds: the ReLU
    # stack saturates in the first epoch and train loss sticks at ~0.26 forever,
    # i.e. the model just predicts a constant. 0.003 trained cleanly on every
    # seed tried (0-3). See --collapse_loss below for the guard.
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--collapse_loss', default=0.25, type=float,
                        help="abort if train loss is still above this after "
                             "--collapse_after epochs, which means the net has died")
    parser.add_argument('--collapse_after', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_frac', default=0.15, type=float,
                        help="fraction of TRAINING speakers held out to choose the "
                             "checkpoint; the test split is scored once at the end")
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    if os.path.abspath(args.train_X) == os.path.abspath(args.test_X):
        parser.error("--train_X and --test_X point at the same file.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    Xall, Yall, names = load_xy(args.train_X, args.train_Y)
    keys = [corpus_speaker(n) for n in names]
    va_mask = holdout_mask(keys, args.val_frac, args.seed)
    describe("cnn", [k for k, v in zip(keys, va_mask) if not v],
             [k for k, v in zip(keys, va_mask) if v])

    Xtrain, Ytrain = Xall[~va_mask], Yall[~va_mask]
    Xval, Yval = Xall[va_mask], Yall[va_mask]
    Xtest, Ytest, _ = load_xy(args.test_X, args.test_Y)
    print(f"train {Xtrain.shape}  val {Xval.shape}  test {Xtest.shape}")
    print(f"annotated per formant - train {(Ytrain > 0).sum(axis=0).tolist()}, "
          f"test {(Ytest > 0).sum(axis=0).tolist()}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="asr-formants")
        run.config.update({"lr": args.lr, "batch_size": args.batch_size,
                           "epochs": args.epochs, "arch": args.arch, "model": "CNN"})

    model = build_model(args.arch).to(device)
    print(f"arch={args.arch}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
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

        epoch_loss = cost / n_batches
        if epoch == args.collapse_after and epoch_loss > args.collapse_loss:
            raise SystemExit(
                f"Training collapsed: train loss {epoch_loss:.4f} after {epoch + 1} epochs "
                f"is above --collapse_loss {args.collapse_loss}. The ReLU stack has "
                f"saturated and the model is predicting a constant. Retry with a lower "
                f"--lr or a different --seed.")

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
