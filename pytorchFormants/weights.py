"""Resolve model checkpoints, downloading from Hugging Face when needed.

Three of the four checkpoints are small enough to live in the repository. The
CNN estimator is not -- its first dense layer alone holds 53.5M parameters, so
the file is ~215MB and exceeds GitHub's 100MB per-file limit. Anything missing
locally is fetched from the Hub and cached.

Override the repo with the DEEPFORMANTS_HF_REPO environment variable, or point
any script at an explicit file with its --model flag to bypass this entirely.
"""
from __future__ import absolute_import, print_function

import os

HF_REPO = os.environ.get("DEEPFORMANTS_HF_REPO", "MLSpeech/DeepFormants")

HERE = os.path.dirname(os.path.abspath(__file__))

# backend -> (filename on the Hub, path inside this repo if it is committed)
CHECKPOINTS = {
    'ff':   ("LPC_NN_scaledLoss.pt", os.path.join(HERE, "Estimator", "LPC_NN_scaledLoss.pt")),
    'cnn':  ("CNN_estimate.pt",      os.path.join(HERE, "Estimator", "CNN_estimate.pt")),
    'rnn':  ("LPC_RNN.pt",           os.path.join(HERE, "Tracker", "LPC_RNN.pt")),
    'crnn': ("CRNN_tracker.pt",      os.path.join(HERE, "CRNN", "CRNN_tracker.pt")),
}


def download(filename, repo=None):
    """Fetch one checkpoint from the Hub, returning the cached local path."""
    repo = repo or HF_REPO
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is needed to download checkpoints that are not in the "
            "repository. Install it with 'pip install huggingface_hub', or pass an "
            "explicit path with --model.")
    return hf_hub_download(repo_id=repo, filename=filename)


def resolve(backend, model_path=None, repo=None):
    """Return a usable checkpoint path for a backend.

    An explicit model_path always wins. Otherwise a committed copy is used when
    present, and the Hub is consulted only as a fallback.
    """
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return model_path

    if backend not in CHECKPOINTS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {sorted(CHECKPOINTS)}")

    filename, local = CHECKPOINTS[backend]
    if os.path.exists(local):
        return local

    print(f"{filename} is not in the repository, fetching from {repo or HF_REPO}...")
    return download(filename, repo)
