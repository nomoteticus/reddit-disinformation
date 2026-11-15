"""Selecting and tidying scraped submissions and comments (Sections 3.2 and 3.3).

Nothing in this module talks to an API, so it can be used and tested offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta

import pandas as pd

SUBMISSION_FIELDS = [
    "id", "title", "subreddit", "author", "url", "domain",
    "is_self", "is_video", "is_crosspostable", "post_hint",
    "num_comments", "score", "removed_by_category",
    "selftext", "link_flair_text", "full_link",
]  # fmt: skip
COMMENT_FIELDS = [
    "id", "created", "subreddit", "link_id", "parent_id", "body", "edited",
    "author", "collapsed", "no_follow", "score", "distinguished",
]  # fmt: skip

#: Posts linking to these domains are not news links (images, videos, Reddit itself).
NON_NEWS_DOMAIN_REGEX = r"reddit|redd.it|imgur|gfycat|google|twitter|youtu|^self\.|^i\.redd"
NON_LINK_HINT_REGEX = "image|video|self"

#: Comments kept per submission.
MAX_COMMENTS_PER_SUBMISSION = 1000


def to_datetime(utc_timestamp, hours_back: int = 0) -> datetime:
    """Convert a Unix timestamp to a datetime."""
    return datetime.fromtimestamp(int(utc_timestamp)) - timedelta(hours=hours_back)


def collapse_whitespace(text: str) -> str:
    """Replace every run of whitespace, including line breaks, by one space."""
    return " ".join(text.split())


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    """Split a sequence into consecutive pieces of at most ``size`` items."""
    return (items[start : start + size] for start in range(0, len(items), size))


def is_news_link(submission) -> bool:
    """Whether a submission shares a link to an external website and has comments."""
    return (
        not submission.is_self
        and not re.search(NON_LINK_HINT_REGEX, str(submission.post_hint))
        and not re.search(NON_NEWS_DOMAIN_REGEX, str(submission.domain))
        and submission.num_comments > 0
    )


def news_link_ids(submissions: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.Series:
    """Ids of the submissions whose comments should be collected."""
    ids = pd.Series(
        [row.id for row in submissions.itertuples() if is_news_link(row)], dtype="object"
    )
    return ids[~ids.isin(set(exclude))]


def clean_comments(comments: pd.DataFrame, columns: list[str] = COMMENT_FIELDS) -> pd.DataFrame:
    """Keep the earliest 1,000 comments per submission and tidy the fields."""
    if comments.empty:
        return comments
    comments = (
        comments[columns]
        .reset_index()
        .sort_values(["link_id", "created"])
        .groupby("link_id")
        .head(MAX_COMMENTS_PER_SUBMISSION)
    )
    return comments.assign(
        created=comments["created"].apply(to_datetime),
        body=comments["body"].apply(collapse_whitespace),
        id="t1_" + comments["id"],
    )


def first_and_second_level(comments: list[dict]) -> list[dict]:
    """Replies to the submission, followed by direct replies to those."""
    first = [comment for comment in comments if comment["parent_id"].startswith("t3_")]
    first_ids = {"t1_" + comment["id"] for comment in first}
    second = [comment for comment in comments if comment["parent_id"] in first_ids]
    return first + second


def merge_new_rows(new: pd.DataFrame, old: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Add newly scraped rows to a table; for ids in both, the new values win."""
    merged = new.set_index("id").combine_first(old.set_index("id")).reset_index()
    return merged.sort_values(sort_by, ascending=False)
