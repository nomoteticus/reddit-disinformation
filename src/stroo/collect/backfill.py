"""One-off collection of past months, before the daily job took over.

``backfill_submissions`` downloads every submission of the tracked subreddits
for the months of a year that have no ``SUBM_yyyy_mm.csv`` yet;
``backfill_comments`` then downloads the comments of the link posts in each
submission file that are not yet in the matching ``COMM_yyyy_mm.csv``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import settings
from ..aggregate.tables import load_tracked_subreddits
from ..flags.detect import monthly_files
from ..io import read_csv, write_csv
from . import pushshift
from .cleaning import COMMENT_FIELDS, chunked, clean_comments, news_link_ids, to_datetime

log = settings.get_logger("stroo.collect", "subm_comm_realtime.log")

COMMENT_BATCH_SIZE = 100


def _days_ago(moment: datetime, now: datetime) -> int:
    return max((now - moment).days, 0)


def backfill_submissions(year: int, raw_dir: Path | None = None, now: datetime | None = None):
    """Download the months of ``year``, up to the current one, that are still missing.

    The most recent existing month is treated as complete; earlier gaps are filled.
    """
    raw_dir = raw_dir or settings.RAW_DIR
    now = now or datetime.now()
    subreddits = load_tracked_subreddits()["subreddit"]

    existing = [name for name in monthly_files(raw_dir, "SUBM") if name.startswith(f"SUBM_{year}")]
    latest_existing = {int(name[10:12]) for name in existing[-1:]}
    last_month = now.month if now.year == year else 12
    for month in sorted(set(range(1, last_month + 1)) - latest_existing):
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
        scraped = []
        for subreddit in subreddits:
            scraped += pushshift.fetch_submissions(
                subreddit,
                sort="asc",
                after=f"{_days_ago(start, now) + 1}d",
                before=f"{_days_ago(end, now)}d",
            )
            log.info("Finished: %s - %d overall cases", subreddit, len(scraped))
        submissions = pd.DataFrame([submission.d_ for submission in scraped])
        submissions = submissions.assign(
            id="t3_" + submissions["id"],
            created_utc=submissions["created_utc"].apply(to_datetime),
        )
        in_month = (submissions["created_utc"] >= start) & (submissions["created_utc"] < end)
        write_csv(submissions[in_month], raw_dir / f"SUBM_{year}_{month:02d}.csv")
        log.info("Wrote SUBM_%d_%02d.csv", year, month)


def backfill_comments(year: int, raw_dir: Path | None = None) -> None:
    """Download comments for link posts that have none in the COMM files yet."""
    raw_dir = raw_dir or settings.RAW_DIR
    submission_files = [
        name for name in monthly_files(raw_dir, "SUBM") if name.startswith(f"SUBM_{year}")
    ]
    for submission_file in reversed(submission_files):
        comment_path = raw_dir / submission_file.replace("SUBM", "COMM")
        submissions = pd.read_csv(raw_dir / submission_file).sort_values(
            ["subreddit", "created_utc"]
        )
        if comment_path.exists():
            already_scraped = set(read_csv(comment_path)["link_id"])
        else:
            already_scraped = set()
            write_csv(pd.DataFrame(columns=COMMENT_FIELDS), comment_path)

        to_scrape = list(news_link_ids(submissions, exclude=already_scraped))
        # As in the original, a month with 100 or fewer missing submissions is left alone.
        if len(to_scrape) <= COMMENT_BATCH_SIZE:
            continue
        n_comments = 0
        for batch in chunked(to_scrape, COMMENT_BATCH_SIZE):
            comments = clean_comments(pushshift.fetch_comments(batch, limit=1_000_000))
            if not comments.empty:
                comments[COMMENT_FIELDS].to_csv(
                    comment_path, mode="a", index=False, header=False, lineterminator="\n"
                )
                n_comments += len(comments)
            log.info("%s: %d comments so far", submission_file, n_comments)
