"""Train the CRNN formant tracker.

Usage:
    python crnn_train.py --train_data .../Train_crnn.npy --test_data .../Test_crnn.npy

Build the .npy inputs first with crnn_feats.py. Loss and optimizer follow the
rest of the package (L1 / Adagrad at lr 0.01); errors are reported in Hz.
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

from crnn_model import CRNN


def load_split(path):
    """Load the object array written by crnn_feats.process_directory."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found at {path}. Run crnn_feats.py first.")
    return np.load(path, allow_pickle=True)


def make_chunks(entries, seq_len):
    """Cut utterances into fixed-length chunks so batches are rectangular.

    Returns X (N, 1, n_bins, seq_len) and Y (N, seq_len, 4).
    """
    xs, ys = [], []
    for _, labels, spec in entries:
        n_chunks = spec.shape[1] // seq_len
        for c in range(n_chunks):
            lo, hi = c * seq_len, (c + 1) * seq_len
            xs.append(spec[:, lo:hi])
            ys.append(labels[lo:hi])
    if not xs:
        raise ValueError(f"No utterance was long enough for seq_len={seq_len}.")
    X = np.asarray(xs, dtype=np.float32)[:, None, :, :]
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y


def evaluate(model, entries, device):
    """Per-formant mean absolute error in Hz, over whole utterances.

    Evaluating on full utterances (rather than chunks) gives the recurrent
    layers the same context they get at inference time.
    """
    model.eval()
    abs_err = np.zeros(4, dtype=np.float64)
    n_frames = 0
    with torch.no_grad():
        for _, labels, spec in entries:
            if spec.shape[1] == 0:
                continue
            x = torch.from_numpy(spec[None, None, :, :].astype(np.float32)).to(device)
            pred = model(x).squeeze(0).cpu().numpy()
            abs_err += np.abs(pred - labels).sum(axis=0)
            n_frames += labels.shape[0]
    model.train()
    if n_frames == 0:
        raise ValueError("Test split contained no frames.")
    return 1000.0 * abs_err / n_frames  # kHz targets -> Hz error


def main():
    parser = argparse.ArgumentParser(description="Train the CRNN formant tracker.")
    parser.add_argument('--train_data', required=True, help="Train_crnn.npy from crnn_feats.py")
    parser.add_argument('--test_data', required=True, help="Test_crnn.npy from crnn_feats.py")
    parser.add_argument('--output', default="CRNN_tracker.pt", help="where to write the best checkpoint")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--seq_len', default=20, type=int, help="frames per training chunk")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb', action='store_true', help="log to wandb (project asr-formants)")
    args = parser.parse_args()

    if args.train_data == args.test_data:
        parser.error("--train_data and --test_data point at the same file; "
                     "test scores would be measured on training data.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_entries = load_split(args.train_data)
    test_entries = load_split(args.test_data)
    Xtrain, Ytrain = make_chunks(train_entries, args.seq_len)
    print(f"train chunks: {Xtrain.shape}, test utterances: {len(test_entries)}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="asr-formants")
        run.config.update({"lr": args.lr, "batch_size": args.batch_size,
                           "epochs": args.epochs, "seq_len": args.seq_len,
                           "model": "CRNN"})

    model = CRNN(n_bins=Xtrain.shape[2]).to(device)
    if run is not None:
        run.watch(model)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    n_batches = int(np.ceil(Xtrain.shape[0] / args.batch_size))
    best_loss = float('inf')

    print("Starting training")
    for epoch in range(args.epochs):
        order = np.random.permutation(Xtrain.shape[0])
        cost = 0.0
        for b in range(n_batches):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            xb = torch.from_numpy(Xtrain[idx]).to(device)
            yb = torch.from_numpy(Ytrain[idx]).to(device)

            optimizer.zero_grad()
            output = loss_fn(model(xb), yb)
            output.backward()
            optimizer.step()
            cost += output.item()

        errors = evaluate(model, test_entries, device)
        total_loss = float(errors.mean())
        print(f"Epoch {epoch}: train loss {cost / n_batches:.4f} | "
              f"test MAE Hz F1 {errors[0]:.1f} F2 {errors[1]:.1f} "
              f"F3 {errors[2]:.1f} F4 {errors[3]:.1f} | mean {total_loss:.1f}")

        if run is not None:
            run.log({"train_loss": cost / n_batches, "mean_abs_error_hz": total_loss,
                     "F1": errors[0], "F2": errors[1], "F3": errors[2], "F4": errors[3]})

        # Only overwrite the checkpoint when the model actually improved.
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), args.output)
            print(f"  saved {args.output} (mean test MAE {best_loss:.1f} Hz)")

    print(f"Best mean test MAE: {best_loss:.1f} Hz -> {args.output}")


if __name__ == "__main__":
    main()
