"""Thin wrappers around the Pushshift API (via ``psaw``) with error handling.

Status: Pushshift closed its public API in 2023, so these calls no longer work
without moderator access. The module documents how the data of the paper were
collected (Section 3) and is the place to plug in another source.

Requires the optional dependencies ``pip install stroo[collect]``;
``timeout_decorator`` relies on Unix signals and does not work on Windows.
"""

from __future__ import annotations

import time

import pandas as pd
import timeout_decorator
from psaw import PushshiftAPI

from .. import settings
from .cleaning import COMMENT_FIELDS, SUBMISSION_FIELDS, first_and_second_level

log = settings.get_logger("stroo.collect", "subm_comm_realtime.log")

#: Seconds to wait before each request, to stay within the API's rate limit.
PAUSE = 3
COMMENT_TIMEOUT = 60


def fetch_submissions(
    subreddit: str, *, after: str, before: str, limit: int = 1_000_000, sort: str = "desc"
) -> list:
    """All submissions of a subreddit in a time window; an empty list on failure.

    ``after`` and ``before`` are Pushshift relative times such as ``"8d"``.
    """
    time.sleep(PAUSE)
    try:
        return list(
            PushshiftAPI().search_submissions(
                subreddit=subreddit, limit=limit, before=before, after=after, sort=sort,
                lang="en", filter=SUBMISSION_FIELDS,
            )
        )  # fmt: skip
    except Exception as error:  # the API fails in many ways; skip and retry later
        log.error("%s: %s", subreddit, type(error).__name__)
        return []


@timeout_decorator.timeout(COMMENT_TIMEOUT, timeout_exception=StopIteration)
def fetch_comments(
    submission_ids, fields: list[str] = COMMENT_FIELDS, limit: int = 1_000_000
) -> pd.DataFrame:
    """First- and second-level comments of a batch of submissions.

    Returns an empty table if the request fails or takes longer than a minute.
    """
    time.sleep(PAUSE)
    try:
        found = PushshiftAPI().search_comments(
            link_id=",".join(submission_ids), limit=limit, filter=fields
        )
        comments = first_and_second_level([comment.d_ for comment in found])
    except Exception as error:
        log.error("comments: %s", type(error).__name__)
        comments = []
    if not comments:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(comments)
