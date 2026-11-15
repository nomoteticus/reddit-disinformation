"""Comparison of user flags with a fact-checker's list of websites (Section 5.5).

The paper uses the 369 domains listed by NewsGuard's coronavirus
misinformation tracking centre on 29 November 2020. The list itself is not
redistributed here; the post-level table published on Figshare (``SSD.csv``,
doi:10.6084/m9.figshare.13174145) already marks each post's domain as listed
or not in its ``fakedomain`` column.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .. import settings
from ..io import read_csv

#: Posts linking to these are not links to news websites.
NOT_A_WEBSITE_REGEX = r"redd\.it|^self\.|twitter|/r/|youtu|img|gif|giph|blog"


def build_post_table(
    year: int,
    months: Iterable[int],
    suspicious_domains: Iterable[str],
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per scraped post: linked domain, removal, and whether it was flagged.

    A post counts as flagged if any direct reply to it matches a pattern.
    """
    raw_dir = raw_dir or settings.RAW_DIR
    processed_dir = processed_dir or settings.PROCESSED_DIR
    suspicious = set(suspicious_domains)
    per_month = []
    for month in months:
        tag = f"{year}_{month:02d}"
        submissions = read_csv(raw_dir / f"SUBM_{tag}.csv")
        comments = read_csv(raw_dir / f"COMM_{tag}.csv")
        matches = read_csv(processed_dir / f"MATCH_{tag}.csv")

        flags_per_post = (
            comments.rename(columns={"id": "comm_id", "parent_id": "subm_id"})[
                ["subm_id", "comm_id"]
            ]
            .set_index("comm_id")
            .join(matches[["comm_id", "flag"]].set_index("comm_id"))
            .fillna(0)
            .groupby("subm_id")["flag"]
            .sum()
        )
        posts = submissions.rename(columns={"id": "subm_id"}).assign(
            was_removed=submissions["removed_by_category"].notna().astype(int)
        )[["subm_id", "subreddit", "domain", "was_removed"]]
        per_month.append(posts.set_index("subm_id").join(flags_per_post, how="left").fillna(0))

    posts = pd.concat(per_month)
    posts = posts.assign(
        domain=posts["domain"].astype(str),
        isflagged=(posts["flag"] > 0).astype(int),
        fakedomain=posts["domain"].astype(str).isin(suspicious).astype(int),
        total=1,
    )
    return posts[~posts["domain"].str.contains(NOT_A_WEBSITE_REGEX)].reset_index()


def compare_with_suspicious(posts: pd.DataFrame) -> dict[str, float]:
    """The figures quoted in Section 5.5, from a table made by ``build_post_table``."""
    domain = posts["domain"].astype(str)
    suspicious = posts["fakedomain"] > 0
    flagged = posts["isflagged"] > 0

    n_suspicious_domains = domain[suspicious].nunique()
    n_other_domains = domain[~suspicious].nunique()
    posts_suspicious, posts_other = int(suspicious.sum()), int((~suspicious).sum())
    flagged_suspicious, flagged_other = (
        int((suspicious & flagged).sum()),
        int((~suspicious & flagged).sum()),
    )
    share_suspicious = flagged_suspicious / posts_suspicious
    share_other = flagged_other / posts_other
    return {
        "suspicious domains shared on Reddit": n_suspicious_domains,
        "suspicious domains flagged at least once": domain[suspicious & flagged].nunique(),
        "share of suspicious domains flagged": domain[suspicious & flagged].nunique()
        / n_suspicious_domains,
        "share of other domains flagged": domain[~suspicious & flagged].nunique() / n_other_domains,
        "posts from suspicious domains": posts_suspicious,
        "posts from suspicious domains flagged": flagged_suspicious,
        "share of posts from suspicious domains flagged": share_suspicious,
        "share of posts from other domains flagged": share_other,
        "odds ratio": (share_suspicious / (1 - share_suspicious))
        / (share_other / (1 - share_other)),
    }


def suspicious_domains_per_subreddit(posts: pd.DataFrame) -> pd.Series:
    """Number of distinct suspicious domains shared in each subreddit."""
    suspicious = posts[posts["fakedomain"] > 0]
    return suspicious.groupby("subreddit")["domain"].nunique().sort_values(ascending=False)
