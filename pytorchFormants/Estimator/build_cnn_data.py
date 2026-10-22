"""Build CNN-estimator datasets from VTR, Hillenbrand and Clopper.

This replaces the lost CNN_ALL/ builder that produced norm_merged_*.npy for
old/Estimator/cnn_freq_norm_stableY.py. One vowel segment becomes one 55x50
spectrogram image plus F1-F4 in kHz.

Annotation density differs by corpus and is handled uniformly:

  * VTR         - every frame carries F1-F4
  * Hillenbrand - F1-F3 annotated at a few points per vowel, F4 mostly absent
  * Clopper     - only F1 and F2, annotated at the vowel midpoint

A segment's label for a formant is the mean over the frames where that formant
is actually annotated, and -1 when it never is. Training masks -1 out of the
loss, so Clopper contributes F1/F2 gradients only rather than dragging F3/F4
towards a nonsense target.

Usage:
    python build_cnn_data.py --corpora vtr hillenbrand clopper --output_dir data/cnn/
"""
from __future__ import absolute_import, print_function

import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spec_features import SAMPLE_RATE, estimation_image, read_wav

FRAME_SEC = 0.01

VOWELS = {'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'er', 'ey',
          'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux'}

CORPORA = {
    'vtr': {'train': 'vtr/shua_processed/Train', 'test': 'vtr/shua_processed/Test'},
    # Hillenbrand and Clopper ship as a single pool, split by speaker below.
    'hillenbrand': {'all': 'hillenbrand/processed'},
    'clopper': {'all': 'clopper/processed'},
}


def read_label_rows(label_file):
    rows = []
    with open(label_file, 'r') as f:
        for r in csv.reader(f, delimiter="\t"):
            if not r:
                continue
            vals = [float(v) for v in r[1:5]]
            vals += [-1.0] * (4 - len(vals))
            rows.append((r[0], vals))
    return rows


def vowel_segments(rows):
    """Maximal contiguous runs of the same vowel phone -> (phone, start, end)."""
    segs, cur = [], None
    for i, (phone, _) in enumerate(rows):
        if phone in VOWELS:
            if cur is not None and cur[0] == phone:
                cur[2] = i + 1
            else:
                if cur is not None:
                    segs.append(cur)
                cur = [phone, i, i + 1]
        else:
            if cur is not None:
                segs.append(cur)
                cur = None
    if cur is not None:
        segs.append(cur)
    return segs


def segment_label(rows, start, end):
    """Per-formant mean over annotated frames; -1 where never annotated."""
    out = np.full(4, -1.0, dtype=np.float32)
    for k in range(4):
        vals = [rows[i][1][k] for i in range(start, end) if rows[i][1][k] > 0]
        if vals:
            out[k] = float(np.mean(vals)) / 1000.0
    return out


def process_tree(input_dir, tag):
    """Walk a corpus tree, returning (names, X, Y)."""
    names, xs, ys = [], [], []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if not file.endswith(".wav"):
                continue
            wav_file = os.path.join(root, file)
            label_file = os.path.splitext(wav_file)[0] + ".label"
            if not os.path.exists(label_file):
                continue

            rows = read_label_rows(label_file)
            if not rows:
                continue
            data = read_wav(wav_file)
            stem = '_'.join(wav_file.replace('.wav', '').split('/')[-3:])

            for k, (phone, a, b) in enumerate(vowel_segments(rows)):
                label = segment_label(rows, a, b)
                if not (label > 0).any():
                    continue  # nothing annotated in this segment
                seg = data[int(a * FRAME_SEC * SAMPLE_RATE):int(b * FRAME_SEC * SAMPLE_RATE)]
                img = estimation_image(seg)
                if img is None:
                    continue
                names.append(f"{tag}_{stem}_{phone}_{k}")
                xs.append(img)
                ys.append(label)

    if not xs:
        return [], np.zeros((0, 1, 55, 50), np.float32), np.zeros((0, 4), np.float32)
    X = np.asarray(xs, dtype=np.float32)[:, None, :, :]
    return names, X, np.asarray(ys, dtype=np.float32)


def speaker_of(name):
    """Speaker id used to split the single-pool corpora without leakage."""
    stem = name.split('_')[-3] if name.count('_') >= 3 else name
    return stem[:3]


def split_by_speaker(names, X, Y, test_frac=0.2, seed=0):
    speakers = sorted({speaker_of(n) for n in names})
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n_test = max(1, int(round(len(speakers) * test_frac)))
    test_spk = set(speakers[:n_test])
    mask = np.array([speaker_of(n) in test_spk for n in names])
    return (~mask), mask


def main():
    parser = argparse.ArgumentParser(description="Build CNN estimator datasets.")
    parser.add_argument('--base', default="/mlspeech/datasets/public/formants/")
    parser.add_argument('--corpora', nargs='+', default=['vtr'],
                        choices=sorted(CORPORA), help="corpora to include")
    parser.add_argument('--output_dir', default="data/cnn/")
    parser.add_argument('--prefix', default=None,
                        help="output name prefix (default: joined corpus names)")
    parser.add_argument('--test_frac', default=0.2, type=float,
                        help="held-out speaker fraction for corpora with no official split")
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix or '_'.join(args.corpora)

    tr_names, tr_X, tr_Y = [], [], []
    te_names, te_X, te_Y = [], [], []

    for corpus in args.corpora:
        spec = CORPORA[corpus]
        if 'train' in spec:
            for split, acc_n, acc_x, acc_y in (('train', tr_names, tr_X, tr_Y),
                                               ('test', te_names, te_X, te_Y)):
                n, x, y = process_tree(os.path.join(args.base, spec[split]), corpus)
                print(f"{corpus} {split}: {len(n)} segments")
                acc_n.extend(n); acc_x.append(x); acc_y.append(y)
        else:
            n, x, y = process_tree(os.path.join(args.base, spec['all']), corpus)
            tr_mask, te_mask = split_by_speaker(n, x, y, args.test_frac, args.seed)
            n = np.array(n)
            print(f"{corpus}: {len(n)} segments -> {tr_mask.sum()} train / {te_mask.sum()} test "
                  f"(speaker-disjoint)")
            tr_names.extend(n[tr_mask].tolist()); tr_X.append(x[tr_mask]); tr_Y.append(y[tr_mask])
            te_names.extend(n[te_mask].tolist()); te_X.append(x[te_mask]); te_Y.append(y[te_mask])

    for split, names, Xs, Ys in (('train', tr_names, tr_X, tr_Y), ('test', te_names, te_X, te_Y)):
        X = np.concatenate(Xs); Y = np.concatenate(Ys)
        np.save(os.path.join(args.output_dir, f"{prefix}_{split}_X.npy"), X)
        np.save(os.path.join(args.output_dir, f"{prefix}_{split}_Y.npy"),
                np.array([[n] + v.tolist() for n, v in zip(names, Y)], dtype=object),
                allow_pickle=True)
        annotated = (Y > 0).sum(axis=0)
        print(f"{split}: X {X.shape}  annotated per formant {annotated.tolist()}")


if __name__ == "__main__":
    main()
