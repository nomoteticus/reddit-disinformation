"""Drawing the comments that were annotated by hand (Section 4.4.1, Table 2).

For each of the six flag types, comments containing its keyword are sampled:
some from all keyword matches (``sample == "all"``) and, optionally, some more
from those that also match a POS pattern (``sample == "pos"``).

The three annotated sets of the paper correspond to::

    draw_samples(candidates(2020, months=range(1, 8)), **SETS_1_AND_2)   # January-July
    draw_samples(candidates(2020, months=range(8, 11)), **SET_3)         # August-October
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .. import settings
from ..flags import FLAG_TYPES, LABELS
from ..flags.filters import (
    ABOUT_FAKE_NEWS_REGEX,
    BOT_AUTHOR_REGEX,
    BOT_BODY_REGEX,
    SARCASM_REGEX,
)
from ..io import read_csv, write_csv

SEED = 777
#: 100 comments per type from keyword matches + 100 per type from POS matches.
SETS_1_AND_2 = {"n_keyword": 100, "n_pos": 100}
#: 50 comments per type from keyword matches only.
SET_3 = {"n_keyword": 50, "n_pos": 0}
OUTPUT_COLUMNS = ["word", "subm_title", "domain", "comm_body", "class", *FLAG_TYPES, "sample",
                  "flag"]  # fmt: skip


def candidates(
    year: int,
    months: Iterable[int],
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Distinct comment texts that can be sampled, with their keyword counts.

    Only first-level comments are kept. Bots, sarcastic comments and posts
    that are themselves about false information are excluded, and so are the
    5% longest comments.
    """
    raw_dir = raw_dir or settings.RAW_DIR
    processed_dir = processed_dir or settings.PROCESSED_DIR
    per_month = []
    for month in months:
        tag = f"{year}_{month:02d}"
        matches = read_csv(processed_dir / f"MATCH_{tag}.csv")
        comments = read_csv(raw_dir / f"COMM_{tag}.csv")
        submissions = read_csv(raw_dir / f"SUBM_{tag}.csv")

        comments["body"] = comments["body"].astype(str)
        is_bot = comments["body"].str.contains(BOT_BODY_REGEX) | comments[
            "author"
        ].str.lower().str.contains(BOT_AUTHOR_REGEX)
        comments = comments[~is_bot].rename(
            columns={"id": "comm_id", "parent_id": "subm_id", "body": "comm_body"}
        )

        titles = submissions["title"].astype(str)
        about_fake_news = titles.str.lower().str.contains(ABOUT_FAKE_NEWS_REGEX)
        submissions = submissions[~about_fake_news].rename(
            columns={"id": "subm_id", "title": "subm_title"}
        )[["subm_id", "subm_title", "domain"]]

        # Joining on the comment's parent keeps direct replies to a submission only.
        joined = (
            matches.groupby("comm_id")[LABELS].sum()
            .join(comments[["comm_id", "comm_body", "subm_id"]].set_index(["subm_id", "comm_id"]),
                  how="inner")
            .loc[lambda table: ~table["comm_body"].str.contains(SARCASM_REGEX)]
            .join(submissions.set_index("subm_id"), how="inner")
            .reset_index()
        )  # fmt: skip
        per_month.append(joined)

    comments = pd.concat(per_month)
    by_text = comments.groupby("comm_body")
    distinct = by_text[LABELS].sum().join(by_text[["subm_title", "domain"]].first()).reset_index()
    distinct["nchar"] = distinct["comm_body"].str.len()
    return distinct[distinct["nchar"] < distinct["nchar"].quantile(0.95)]


def draw_samples(candidates: pd.DataFrame, n_keyword: int, n_pos: int = 0) -> pd.DataFrame:
    """Sample ``n_keyword`` (+ ``n_pos``) comments per flag type.

    A comment's chance of being drawn is proportional to how many of its
    sentences contain the keyword.

    NumPy's global random state is re-seeded before every draw, as in the
    original script, so that the published samples can be drawn again.
    """
    samples = []
    for flag_type in FLAG_TYPES:
        pool = candidates[candidates[flag_type] > 0].assign(word=flag_type, **{"class": ""})

        np.random.seed(SEED)  # noqa: NPY002 - reproduces the published sample
        weights = pool[flag_type] / pool[flag_type].sum()
        drawn = np.random.choice(pool.index, size=n_keyword, replace=False, p=weights)  # noqa: NPY002
        sample = pool.loc[drawn].assign(sample="all")

        if n_pos > 0:
            rest = pool.loc[pool.index.difference(drawn)].query("flag > 0")
            np.random.seed(SEED)  # noqa: NPY002
            weights = rest[flag_type] / rest[flag_type].sum()
            drawn = np.random.choice(rest.index, size=n_pos, replace=False, p=weights)  # noqa: NPY002
            np.random.seed(SEED)  # noqa: NPY002
            sample = pd.concat([sample, pool.loc[drawn].assign(sample="pos")]).sample(frac=1)
        samples.append(sample)
    return pd.concat(samples)[OUTPUT_COLUMNS]


def write_annotation_files(samples: pd.DataFrame, folder: Path, name: str) -> None:
    """Write the sample twice: for the coders (blind) and with the matcher's results."""
    folder.mkdir(parents=True, exist_ok=True)
    blind = samples[["word", "subm_title", "domain", "comm_body", "class"]]
    write_csv(blind.reset_index(), folder / f"{name}.csv")
    write_csv(samples.reset_index(), folder / f"{name}_labeled.csv")
