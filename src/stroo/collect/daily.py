"""Daily collection of new submissions and comments from the tracked subreddits.

Run once a day (Section 3 of the paper). It appends to the monthly files
``SUBM_yyyy_mm.csv`` and ``COMM_yyyy_mm.csv`` in ``data/raw``. Because the
number of comments and the removal status of a post change after it is
published, each run goes back ``DAYS_BEHIND`` days before the last scraped
submission and overwrites what it finds.

Failed subreddits and comment batches are retried, with longer time limits and
smaller batches on each round.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import timeout_decorator

from .. import settings
from ..aggregate.tables import load_tracked_subreddits
from ..flags.detect import monthly_files
from ..io import read_csv, write_csv
from . import pushshift
from .cleaning import (
    COMMENT_FIELDS,
    chunked,
    clean_comments,
    collapse_whitespace,
    merge_new_rows,
    news_link_ids,
    to_datetime,
)

log = settings.get_logger("stroo.collect", "subm_comm_realtime.log")

#: "We always scrape three days behind every day" (Section 3.2).
DAYS_BEHIND = 3
SUBMISSION_ROUNDS = 5
#: Submissions per comment request, per retry round.
COMMENT_BATCH_SIZES = [200, 100, 100, 50, 10, 5]


def _month_file(kind: str, year: int, month: int) -> str:
    return f"{kind}_{year}_{month:02d}.csv"


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year, month - 1) if month > 1 else (year - 1, 12)


def scrape_submissions(subreddits, days_back: int) -> list:
    """Scrape every subreddit, retrying the ones that returned nothing."""
    remaining, found = set(subreddits), []
    for round_number in range(1, SUBMISSION_ROUNDS + 1):
        if not remaining:
            break
        fetch = timeout_decorator.timeout(min(round_number, 4) * 60)(pushshift.fetch_submissions)
        log.debug("Round %d", round_number)
        for subreddit in sorted(remaining):
            new = fetch(subreddit, sort="asc", before="10m", after=f"{days_back}d")
            found += new
            log.debug("Finished: %30s. Cases: %5d. Overall: %6d", subreddit, len(new), len(found))
        remaining -= {submission.subreddit for submission in found}
    if remaining:
        log.warning("Subreddits not scraped: %d: %s", len(remaining), sorted(remaining))
    return found


def scrape_comments(submission_ids: pd.Series) -> pd.DataFrame:
    """Scrape comments in batches, retrying missing submissions with smaller batches."""
    comments = pd.DataFrame(columns=COMMENT_FIELDS)
    remaining = submission_ids
    for round_number, batch_size in enumerate(COMMENT_BATCH_SIZES, start=1):
        if len(remaining) == 0:
            break
        log.debug("Round %d", round_number)
        for batch in chunked(remaining, batch_size):
            comments = pd.concat([comments, pushshift.fetch_comments(batch)])
            log.debug("%10d comments so far", len(comments))
        remaining = remaining[~remaining.isin(set(comments["link_id"]))]
    if len(remaining) > 0:
        log.warning("Submissions not scraped: %d", len(remaining))
    return clean_comments(comments)


def _update_month(new: pd.DataFrame, path: Path, sort_by: str, template: pd.DataFrame) -> None:
    """Merge ``new`` rows into a monthly file (created if missing)."""
    if new.empty:
        log.info("Nothing to add to %s", path.name)
        return
    old = read_csv(path, parse_dates=[sort_by]) if path.exists() else template.iloc[:0]
    write_csv(merge_new_rows(new, old, sort_by=sort_by), path)
    log.info("Wrote %d rows to %s", len(new), path.name)


def run(raw_dir: Path | None = None, now: datetime | None = None) -> None:
    """Collect new submissions and their comments and update the monthly files."""
    raw_dir = raw_dir or settings.RAW_DIR
    now = now or datetime.now()
    current = (now.year, now.month)
    previous = _previous_month(*current)

    subreddits = load_tracked_subreddits()["subreddit"]
    log.info("Tracked subreddits: %d", len(subreddits))

    # Where did the last run stop? Look in this month's file, else last month's.
    existing = monthly_files(raw_dir, "SUBM")
    last_file = _month_file("SUBM", *(current if _month_file("SUBM", *current) in existing
                                      else previous))  # fmt: skip
    last_submissions = read_csv(raw_dir / last_file, parse_dates=["created_utc"])
    days_since_last = (now - last_submissions["created_utc"].max()).days
    log.info("Days since last submission scraped: %d", days_since_last)

    # --- Submissions
    scraped = scrape_submissions(subreddits, days_back=days_since_last + DAYS_BEHIND)
    if not scraped:
        log.warning("No new submissions; nothing to do")
        return
    submissions = pd.DataFrame([submission.d_ for submission in scraped])
    submissions = submissions.assign(
        id="t3_" + submissions["id"],
        created_utc=submissions["created_utc"].apply(to_datetime),
        title=submissions["title"].astype(str).apply(collapse_whitespace),
    )[last_submissions.columns]

    by_month = {}
    for year, month in (current, previous):
        created = submissions["created_utc"].dt
        by_month[year, month] = submissions[(created.year == year) & (created.month == month)]
        _update_month(
            by_month[year, month], raw_dir / _month_file("SUBM", year, month),
            "created_utc", last_submissions,
        )  # fmt: skip

    # --- Comments, only for submissions that share a link to a website
    to_scrape = news_link_ids(submissions)
    log.info("Submissions to scrape for comments: %d", len(to_scrape))
    comments = scrape_comments(to_scrape)
    for (year, month), month_submissions in by_month.items():
        _update_month(
            comments[comments["link_id"].isin(set(month_submissions["id"]))],
            raw_dir / _month_file("COMM", year, month),
            "created", comments,
        )  # fmt: skip
    log.info("Finished")
