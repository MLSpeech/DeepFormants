"""Speaker-disjoint validation splits carved out of the training data.

Model selection must not touch the test set. Every training script here holds
out a slice of its *training* data for choosing the checkpoint, and the test
split is scored once at the end with the already-chosen weights.

Splits are disjoint by speaker rather than by row, so a speaker's frames never
appear on both sides and the validation score is not inflated by speaker
memorisation.
"""
from __future__ import absolute_import, print_function

import numpy as np

VTR_UNKNOWN = "<unknown>"


def vtr_speaker(name):
    """Speaker id from a VTR row name.

    Names look like dr3_mbef0_sx201_frame_0 or dr3_mbef0_sx201_ax_0, and the
    second field is the speaker.
    """
    parts = str(name).split('_')
    return parts[1] if len(parts) > 1 else VTR_UNKNOWN


def holdout_mask(keys, frac=0.15, seed=0):
    """Boolean mask marking a held-out subset, disjoint by key.

    keys is one grouping key per item (typically a speaker id). Returns a mask
    that is True for held-out items.
    """
    keys = [str(k) for k in keys]
    unique = sorted(set(keys))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct speakers to hold out a validation set.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_val = max(1, int(round(len(unique) * frac)))
    val = {unique[i] for i in order[:n_val]}
    return np.array([k in val for k in keys], dtype=bool)


def split_rows(names, frac=0.15, seed=0, key_fn=vtr_speaker):
    """Train/validation masks for a flat row array addressed by name."""
    val = holdout_mask([key_fn(n) for n in names], frac, seed)
    return ~val, val


def split_items(items, frac=0.15, seed=0, key_fn=None):
    """Split a list of items into (train, val) lists, disjoint by speaker.

    key_fn maps an item to its grouping key; it defaults to reading the
    speaker out of item[0], which matches the (name, ...) tuples used by the
    CRNN and attention loaders.
    """
    if key_fn is None:
        def key_fn(item):
            return vtr_speaker(item[0])
    val = holdout_mask([key_fn(i) for i in items], frac, seed)
    train_items = [it for it, v in zip(items, val) if not v]
    val_items = [it for it, v in zip(items, val) if v]
    return train_items, val_items


def describe(name, train_keys, val_keys):
    """One line confirming the split really is disjoint."""
    tr, va = set(map(str, train_keys)), set(map(str, val_keys))
    overlap = tr & va
    print(f"{name}: {len(tr)} train speakers, {len(va)} val speakers, "
          f"overlap {len(overlap)}")
    if overlap:
        raise AssertionError(f"Validation split leaks speakers: {sorted(overlap)[:5]}")
