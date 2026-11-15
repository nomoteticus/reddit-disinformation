"""Deterministic stand-in for the Universal Sentence Encoder (tests only).

Each text gets a 512-dimensional vector made of eight fixed directions with
clearly different weights plus a little noise, so that a PCA of the result has
well-separated components and is numerically stable across library versions.
Written to run on Python 3.8 / NumPy 1.x as well.
"""

import hashlib

import numpy as np

_DIRECTIONS = np.linalg.qr(np.random.RandomState(0).normal(size=(512, 8)))[0].T  # (8, 512)
_WEIGHTS = np.array([40.0, 28.0, 20.0, 14.0, 10.0, 2.0, 1.5, 1.0])


def fake_encode(texts):
    out = np.empty((len(texts), 512))
    for i, text in enumerate(texts):
        seed = int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        out[i] = (rng.normal(size=8) * _WEIGHTS) @ _DIRECTIONS + 0.01 * rng.normal(size=512)
    return out
