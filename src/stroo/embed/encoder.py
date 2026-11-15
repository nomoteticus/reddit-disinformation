"""Sentence embeddings of post titles, reduced with PCA (Section 7.3).

Titles are encoded with the Universal Sentence Encoder (512 dimensions) and
reduced to a handful of components, which the dashboard uses for clustering
and for finding similar posts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import settings

EMBEDDING_SIZE = 512
BATCH_SIZE = 10_000

Encoder = Callable[[Sequence[str]], np.ndarray]


def load_encoder(model_dir: Path | None = None) -> Encoder:
    """Load the Universal Sentence Encoder (v4) from a local SavedModel folder.

    TensorFlow is imported here, not at module level, so the rest of the
    package works without it. Download the model from
    https://tfhub.dev/google/universal-sentence-encoder/4 and unpack it into
    ``data/models/universal-sentence-encoder_4``.
    """
    import tensorflow_hub as hub

    model_dir = model_dir or settings.MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Universal Sentence Encoder not found at {model_dir}. See the README "
            "section 'Embeddings' for how to obtain it."
        )
    model = hub.load(str(model_dir))
    return lambda texts: model(list(texts)).numpy()


def embed_titles(titles: pd.Series, encode: Encoder) -> np.ndarray:
    """Encode titles in batches; returns an array of shape (n_titles, 512)."""
    n_batches = int(1 + len(titles) / BATCH_SIZE)
    embeddings = np.empty((0, EMBEDDING_SIZE))
    for batch in np.array_split(titles, n_batches):
        embeddings = np.vstack((embeddings, encode(batch)))
    return embeddings


def principal_components(embeddings: np.ndarray, n_components: int = 5) -> pd.DataFrame:
    """Reduce title embeddings to ``n_components`` columns ``PC01``, ``PC02``, ...

    PCA is fitted on the transposed embedding matrix (512 rows, one column
    per title); each title is given its loadings on the components.
    """
    pca = PCA(n_components=n_components).fit(embeddings.T)
    columns = [f"PC{number:02d}" for number in range(1, n_components + 1)]
    return pd.DataFrame(pca.components_.T, columns=columns)


def add_title_components(
    submissions: pd.DataFrame, encode: Encoder, n_components: int = 5
) -> pd.DataFrame:
    """Append the PCA columns of ``subm_title`` to a table of submissions."""
    components = principal_components(embed_titles(submissions["subm_title"], encode), n_components)
    return pd.concat([submissions, components], axis=1)
