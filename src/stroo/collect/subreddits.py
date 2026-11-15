"""Choosing which subreddits to track (Section 3.1 of the paper).

Three steps, repeated weekly while the project ran:

1. ``discover`` - search all of Reddit for posts matching the COVID-19 keywords
   that link to an external website, and record the subreddit of each.
2. ``tabulate`` - count those posts per subreddit and day, keep subreddits with
   more than 10 a day, and look up subscribers, comments per post and the share
   of removed posts. New subreddits are appended to ``config/subreddits.csv``.
3. Categorise the new rows by hand in that file (category, language), then run
   ``apply_keep_rule`` to set the ``keep`` column.

Steps 1 and 2 need the Pushshift API, which is no longer publicly available.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import settings
from ..io import read_csv, write_csv
from .cleaning import to_datetime

log = settings.get_logger("stroo.subreddits", "subr_tabulated.log")

SEARCH_FIELDS = [
    "id", "title", "subreddit", "subreddit_subscribers",
    "author", "full_link", "url", "domain", "is_self",
]  # fmt: skip
MIN_SUBSCRIBERS_SEARCH = 1000
#: Categories that are tracked; entertainment and the rest are discarded.
TRACKED_CATEGORIES = [
    "generic_news", "generic_politics", "coronavirus", "local", "ideological",
    "science_health", "technology", "business_economics",
]  # fmt: skip

# Thresholds of the selection rule.
MIN_SUBSCRIBERS_K = 10
MIN_DAILY_POSTS = 10
MIN_COMMENTS_PER_POST = 1


def _work_dir() -> Path:
    folder = settings.PROCESSED_DIR / "subreddits"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def covid_search_string(keywords_file: Path | None = None) -> str:
    """The COVID-19 keywords joined into one Pushshift query (``corona*|covid*|...``)."""
    with open(keywords_file or settings.COVID_KEYWORDS_FILE, encoding="utf-8") as file:
        return "|".join(line.rstrip() for line in file)


def discover(days_back: int = 1, limit: int = 100_000, past: bool = False) -> pd.DataFrame:
    """Search Reddit for COVID-19 link posts and store them with their subreddit.

    The daily run (``past=False``) appends posts of the last day to
    ``covid_posts.csv``. The initial run (``past=True``, 60 days back in the
    original) writes ``covid_posts_past.csv`` and filters fewer link domains.
    """
    from psaw import PushshiftAPI

    path = _work_dir() / ("covid_posts_past.csv" if past else "covid_posts.csv")
    known = read_csv(path) if path.exists() and not past else pd.DataFrame()
    known_ids = set(known["id"]) if not known.empty else set()
    excluded = (
        "reddit|redd.it|imgur|gfycat" if past
        else r"reddit|redd.it|imgur|gfycat|google|twitter|youtu|^self\.|^i\.redd"
    )  # fmt: skip

    found, n_errors = [], 0
    results = PushshiftAPI().search_submissions(
        q=covid_search_string(), limit=limit, after=f"{days_back}d", sort="desc",
        filter=SEARCH_FIELDS,
    )  # fmt: skip
    for post in results:
        try:
            if (
                post.id not in known_ids
                and post.subreddit_subscribers > MIN_SUBSCRIBERS_SEARCH
                and not post.is_self
                and not re.search(excluded, post.domain)
            ):
                found.append(post.d_)
        except AttributeError:  # posts with missing fields
            n_errors += 1
    log.info("Found %d new posts (%d skipped for missing fields)", len(found), n_errors)

    posts = pd.DataFrame(found)
    if not posts.empty:
        posts["date_added"] = datetime.now()
        posts["created_utc"] = posts["created_utc"].apply(to_datetime)
        if past:
            # Hard-coded in the original: the first run kept posts from March 2020 on.
            posts = posts[posts["created_utc"] >= "2020-03-01"]
    if past or known.empty:
        write_csv(posts, path)
    else:
        posts.to_csv(path, index=False, header=False, mode="a", lineterminator="\n")
    return posts


def posts_per_day(posts: pd.DataFrame) -> pd.DataFrame:
    """Subreddits by days: number of COVID-19 link posts, plus two averages.

    ``total`` is the daily average over the whole period and ``total7d`` over
    the last seven days. Only subreddits above 10 posts a day
    on either average are returned.
    """
    created = pd.to_datetime(posts["created_utc"])
    posts = posts.assign(month_day=created.dt.strftime("%m-%d"))
    counts = (
        posts.groupby(["subreddit", "month_day"])["id"].count().loc[lambda n: n > 1]
        .unstack(level=1).fillna(0).sort_index(axis=1)
    )  # fmt: skip
    n_days = counts.shape[1]
    counts["total"] = counts.iloc[:, :n_days].sum(axis=1) / n_days
    last_week = counts.iloc[:, max(n_days - 7, 0) : n_days]
    counts["total7d"] = last_week.sum(axis=1) / last_week.shape[1]
    busy = (counts["total"] > MIN_DAILY_POSTS) | (counts["total7d"] > MIN_DAILY_POSTS)
    return counts[busy].sort_values("total7d", ascending=False)


def fetch_metadata(subreddits, rounds: int = 5) -> pd.DataFrame:
    """Subscribers, mean comments per visible post and share of removed posts.

    Based on up to 1,000 posts of the last 30 days per subreddit.
    """
    import timeout_decorator
    from psaw import PushshiftAPI

    api = PushshiftAPI()

    @timeout_decorator.timeout(30)
    def recent_posts(subreddit):
        try:
            return [
                post.d_
                for post in api.search_submissions(
                    subreddit=subreddit, limit=1000, after="30d", sort="desc",
                    filter=["subreddit", "subreddit_subscribers", "num_comments",
                            "removed_by_category"],
                )
            ]  # fmt: skip
        except Exception:
            log.error("Error: %s", subreddit)
            return []

    rows, remaining = [], set(subreddits)
    for _ in range(rounds):
        if not remaining:
            break
        for subreddit in sorted(remaining):
            time.sleep(2)
            posts = pd.DataFrame(recent_posts(subreddit))
            if posts.empty:
                continue
            if "removed_by_category" in posts.columns:
                visible = posts["removed_by_category"].isna()
                rows.append((posts["subreddit"][0], posts["subreddit_subscribers"][0],
                             posts["num_comments"][visible].mean(), (~visible).mean()))  # fmt: skip
                remaining.discard(subreddit)
            else:
                rows.append((posts["subreddit"][0], posts["subreddit_subscribers"][0],
                             posts["num_comments"].mean(), 0))  # fmt: skip
    meta = pd.DataFrame(
        rows, columns=["subreddit", "subscribers", "ncomments", "removed"]
    ).set_index("subreddit")
    return pd.DataFrame(
        {
            "subscribers_K": (meta["subscribers"] / 1000).astype(int),
            "avg_ncomments": meta["ncomments"].round(1),
            "perc_removed": (100 * meta["removed"]).round(1),
        }
    )


def tabulate(subreddits_file: Path | None = None) -> pd.DataFrame:
    """Update the per-subreddit activity table and add new subreddits to the config."""
    subreddits_file = subreddits_file or settings.SUBREDDITS_FILE
    work_dir = _work_dir()
    past, present = (
        read_csv(work_dir / "covid_posts_past.csv"),
        read_csv(work_dir / "covid_posts.csv"),
    )
    posts = pd.concat([past, present[~present["id"].isin(set(past["id"]))]]).sort_values(
        "created_utc", ascending=False
    )
    daily = posts_per_day(posts)
    meta = fetch_metadata(daily.index)

    classification = pd.read_csv(subreddits_file).set_index("subreddit")
    table = (
        classification[["keep", "category", "language"]]
        .join(meta, how="right")
        .join(daily.iloc[:, -5:], how="left")
        .round(1)
    )
    write_csv(table.reset_index(), work_dir / "subreddit_activity.csv")

    new = table[table["category"].isna() & ~table.index.isin(classification.index)]
    if not new.empty:
        new.to_csv(subreddits_file, mode="a", header=False, lineterminator="\n")
        log.info("Added %d new subreddits to %s for manual classification",
                 len(new), subreddits_file.name)  # fmt: skip
    return table


def apply_keep_rule(subreddits_file: Path | None = None) -> pd.DataFrame:
    """Set ``keep`` from the selection criteria, after the manual classification.

    A subreddit is tracked if it has more than 10,000 subscribers, more than 10
    COVID-19 link posts a day over the last week, English as its language, more
    than one comment per post on average, and a tracked category.
    """
    subreddits_file = subreddits_file or settings.SUBREDDITS_FILE
    table = pd.read_csv(subreddits_file).sort_values("subreddit").set_index("subreddit")
    table["category"] = table["category"].fillna("")
    table["language"] = table["language"].fillna("en")
    # Number each batch of newly added (still uncategorised) subreddits.
    table.loc[table["category"] == "", "new"] = 1 + np.nanmax(table["new"].fillna(-1))
    table["keep"] = (
        (table["subscribers_K"] > MIN_SUBSCRIBERS_K)
        & (table["total7d"] > MIN_DAILY_POSTS)
        & (table["language"] == "en")
        & (table["avg_ncomments"] > MIN_COMMENTS_PER_POST)
        & table["category"].isin(TRACKED_CATEGORIES)
    )
    table.to_csv(subreddits_file, lineterminator="\n")
    return table
