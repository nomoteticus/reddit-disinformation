"""Join flags, comments and submissions and build the dashboard tables.

Input, for one calendar year (``data/raw`` and ``data/processed``)::

    SUBM_yyyy_mm.csv   submissions        COMM_yyyy_mm.csv   comments
    MATCH_yyyy_mm.csv  matched sentences (from ``stroo.flags.detect``)

Output:

* ``data/processed/UNITED_yyyy_FLAG.csv`` - one row per flag sentence with its
  comment and submission; ``UNITED_yyyy_SUBM/AUTH/DOM.csv`` - counts of all
  submissions per subreddit, author and linked domain (the denominators for
  "percent flagged").
* ``data/dashboard/app_*.csv`` - the six tables the dashboard reads, aggregated
  by flag, submission, subreddit, author (flaggers and flagged) and domain
  (Appendix A of the paper).

The job is incremental. If the UNITED files exist, only the two most recent
months are read and only submissions from the last three weeks are refreshed.

Weeks are ISO week numbers within one year (week 53 is recoded to 0); the
tables are built one year at a time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import settings
from ..embed.encoder import Encoder, add_title_components, load_encoder
from ..flags import FLAG_TYPES
from ..flags.filters import ABOUT_FAKE_NEWS_REGEX, BOT_AUTHOR_REGEX, BOT_BODY_REGEX, COVID_REGEX
from ..io import read_csv, write_csv

log = settings.get_logger("stroo.aggregate", "pos_embed_unite.log")

FLAG_COLUMNS = ["flag"] + FLAG_TYPES + ["other"]

SUBMISSION_COLUMNS = {
    "subreddit": "subreddit",
    "id": "subm_id",
    "title": "subm_title",
    "year": "year",
    "month": "month",
    "week": "week",
    "day": "day",
    "full_link": "subm_link",
    "url": "link_url",
    "domain": "link_domain",
    "removed_by_category": "subm_removed",
    "subm_covid": "subm_covid",
    "subm_fake": "subm_fake",
    "author": "subm_author",
    "num_comments": "subm_ncomments",
}
COMMENT_COLUMNS = {
    "subreddit": "subreddit",
    "parent_id": "comm_parent_id",
    "link_id": "subm_id",
    "id": "comm_id",
    "score": "comm_score",
    "body": "comm_body",
    "author": "comm_author",
}
UNITED_COLUMNS = [
    "year", "month", "week", "day", "subreddit_cat", "sent",
    *FLAG_COLUMNS,
    "subm_title", "subm_removed", "subm_covid", "subm_fake",
    "subm_link", "link_url", "link_domain", "subm_ncomments", "subm_author",
    "comm_parent_id", "comm_body", "comm_author",
]  # fmt: skip

# Grouping keys.
BY_SENTENCE = ["subreddit", "subm_id", "comm_id", "sent_id"]
BY_SUBREDDIT_DAY = ["subreddit_cat", "subreddit", "subm_covid", "year", "month", "week", "day"]
BY_SUBREDDIT_WEEK = ["subreddit_cat", "subreddit", "subm_covid", "week"]
BY_SUBMISSION_DAY = BY_SUBREDDIT_DAY + ["subm_id", "subm_title"]
BY_COMMENT_DAY = BY_SUBMISSION_DAY + ["comm_id", "comm_body"]
BY_AUTHOR_WEEK = BY_SUBREDDIT_WEEK + ["subm_author"]
BY_DOMAIN_WEEK = BY_SUBREDDIT_WEEK + ["link_domain"]

#: A flagger, flagged author or domain needs at least this many rows to be listed.
MIN_ROWS = 2
#: On an incremental run, submissions up to this many weeks back are refreshed.
WEEKS_REFRESHED = 3


# --- Reading and preparing the monthly files --------------------------------


def week_of_year(dates: pd.Series) -> pd.Series:
    """ISO week number, with week 53 recoded to 0."""
    return dates.dt.isocalendar().week.astype("int64").replace(53, 0)


def load_tracked_subreddits(subreddits_file: Path | None = None) -> pd.DataFrame:
    """Subreddits marked ``keep`` in the classification file, with their category."""
    subreddits = pd.read_csv(subreddits_file or settings.SUBREDDITS_FILE)
    kept = subreddits[subreddits["keep"].notna() & subreddits["keep"].astype(bool)]
    return kept[["subreddit", "category"]].rename(columns={"category": "subreddit_cat"})


def prepare_submissions(submissions: pd.DataFrame, subreddits: pd.DataFrame) -> pd.DataFrame:
    """Add topic indicators, calendar fields and the subreddit category."""
    parts = submissions[["title", "subreddit", "selftext"]].fillna("").astype(str)
    text = (parts["title"] + parts["subreddit"] + parts["selftext"]).str.lower()
    created = pd.to_datetime(submissions["created_utc"])
    submissions = submissions.assign(
        subm_covid=text.str.contains(COVID_REGEX),
        subm_fake=text.str.contains(ABOUT_FAKE_NEWS_REGEX),
        removed_by_category=submissions["removed_by_category"].fillna("visible").astype(str),
        year=created.dt.year,
        month=created.dt.month,
        week=week_of_year(created),
        day=created.dt.date,
    )
    return (
        submissions[list(SUBMISSION_COLUMNS)]
        .rename(columns=SUBMISSION_COLUMNS)
        .set_index("subreddit")
        .join(subreddits.set_index("subreddit"), how="left")
        .reset_index()
    )


def prepare_comments(comments: pd.DataFrame) -> pd.DataFrame:
    return comments[list(COMMENT_COLUMNS)].rename(columns=COMMENT_COLUMNS)


def prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ``other``: a pattern matched but none of the six type keywords occurs."""
    is_other = (matches["flag"] > 0) & (matches[FLAG_TYPES].sum(axis=1) == 0)
    return matches.assign(other=is_other.astype(int))


def count_submissions(submissions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Number of submissions, in total and not removed, per group."""
    submissions = submissions.assign(day=submissions["day"].astype(str))
    return submissions.groupby(by).agg(
        n_subm_all=("subm_id", "count"),
        n_subm_kept=("subm_removed", lambda removed: (removed == "visible").sum()),
    )


def join_flags(
    matches: pd.DataFrame, comments: pd.DataFrame, submissions: pd.DataFrame
) -> pd.DataFrame:
    """One row per flag sentence, with its comment and submission attached."""
    united = (
        matches.query("flag > 0")
        .set_index("comm_id")
        .join(comments.set_index("comm_id"), on="comm_id", how="left")
        .reset_index()
        .set_index(BY_SENTENCE)
        .join(submissions.set_index(["subreddit", "subm_id"]), on=["subreddit", "subm_id"],
              how="left")
    )  # fmt: skip
    return united[UNITED_COLUMNS]


def _update(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``new`` take precedence; rows only in ``old`` are kept."""
    return new.combine_first(old) if len(old) > 0 else new


# --- The joined table --------------------------------------------------------


def clean_united(united: pd.DataFrame) -> pd.DataFrame:
    """Final filtering of the joined flag table (step 2 of the pipeline, Section 4.3).

    Removed: flags without a known submission, replies to other comments,
    submissions that are themselves about false information, and bots.
    """
    united = united.reset_index()
    united["day"] = united["day"].astype(str)
    united = united.sort_values("day", ascending=False)
    united = united[~united["day"].isna()]
    united = united[united["comm_parent_id"].str.contains("^t3_")]
    united = united[~united["subm_fake"].astype(bool)]
    is_bot = united["comm_body"].str.lower().str.contains(BOT_BODY_REGEX) | united[
        "comm_author"
    ].str.lower().str.contains(BOT_AUTHOR_REGEX)
    return united[~is_bot]


def build_united(
    year: int, raw_dir: Path, processed_dir: Path, subreddits: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Create or update the UNITED tables for one year.

    Returns the cleaned flag table (``flags``) and the submission counts per
    subreddit, author and domain (``subreddit``, ``author``, ``domain``).
    """
    files = {
        "flags": (processed_dir / f"UNITED_{year}_FLAG.csv", BY_SENTENCE),
        "subreddit": (processed_dir / f"UNITED_{year}_SUBM.csv", BY_SUBREDDIT_WEEK),
        "author": (processed_dir / f"UNITED_{year}_AUTH.csv", BY_AUTHOR_WEEK),
        "domain": (processed_dir / f"UNITED_{year}_DOM.csv", BY_DOMAIN_WEEK),
    }
    months = sorted(
        path.name[-6:-4] for path in raw_dir.glob(f"SUBM_{year}_[0-9][0-9].csv")
        if (raw_dir / path.name.replace("SUBM", "COMM")).exists()
        and (processed_dir / path.name.replace("SUBM", "MATCH")).exists()
    )  # fmt: skip

    try:
        tables = {name: read_csv(path).set_index(index) for name, (path, index) in files.items()}
    except FileNotFoundError:
        tables = {name: pd.DataFrame() for name in files}
        last_week = 0
        log.debug("No UNITED files for %d yet: building from all months", year)
    else:
        months = months[-2:]
        last_week = min(
            tables[name].index.get_level_values("week").max()
            for name in ("subreddit", "author", "domain")
        )
        log.debug("UNITED files found; last complete week: %d", last_week)

    for month in months:
        submissions = read_csv(raw_dir / f"SUBM_{year}_{month}.csv", parse_dates=["created_utc"])
        recent = submissions["created_utc"].dt.isocalendar().week > last_week - WEEKS_REFRESHED
        submissions = submissions[recent]
        if submissions.empty:
            log.info("Nothing to add: SUBM_%d_%s.csv", year, month)
            continue

        submissions = prepare_submissions(submissions, subreddits)
        comments = prepare_comments(read_csv(raw_dir / f"COMM_{year}_{month}.csv"))
        matches = prepare_matches(read_csv(processed_dir / f"MATCH_{year}_{month}.csv"))
        log.info("Updating weeks: %s", sorted(submissions["week"].unique()))

        tables["flags"] = _update(join_flags(matches, comments, submissions), tables["flags"])
        for name, by in [
            ("subreddit", BY_SUBREDDIT_WEEK),
            ("author", BY_AUTHOR_WEEK),
            ("domain", BY_DOMAIN_WEEK),
        ]:
            tables[name] = _update(count_submissions(submissions, by), tables[name])
        log.info("Finished month %s. Flag sentences so far: %d", month, len(tables["flags"]))

    tables["flags"] = clean_united(tables["flags"])

    for name, (path, _) in files.items():
        table = tables[name]
        if name != "flags":
            table = table.reset_index().sort_values("week", ascending=False)
        write_csv(table, path)
    return tables


# --- The dashboard tables -----------------------------------------------------


def _at_least(table: pd.DataFrame, column: str, min_rows: int) -> pd.DataFrame:
    """Keep rows whose value in ``column`` occurs at least ``min_rows`` times."""
    table = table.reset_index()
    counts = table[column].value_counts()
    return table[table[column].isin(counts[counts >= min_rows].index)]


def build_dashboard_tables(
    united: dict[str, pd.DataFrame], encode: Encoder
) -> dict[str, pd.DataFrame]:
    """Aggregate the joined flag table at the levels shown in the dashboard."""
    flags = united["flags"]

    flags_day = flags.reset_index().loc[
        :,
        ["subreddit_cat", "subreddit", "subm_id", "subm_covid", "subm_title", "subm_link",
         "link_url", "subm_removed", "sent_id", "sent", "year", "month", "week", "day",
         *FLAG_COLUMNS],
    ].sort_values("day", ascending=False)  # fmt: skip

    # A comment counts once per flag type, however many of its sentences match.
    comments_day = flags.groupby(BY_COMMENT_DAY)[FLAG_COLUMNS].sum().reset_index()
    comments_day[FLAG_COLUMNS] = comments_day[FLAG_COLUMNS] > 0

    # Submissions: number of flagging comments, plus title embeddings.
    submissions_day = comments_day.groupby(BY_SUBMISSION_DAY)[FLAG_COLUMNS].sum().reset_index()
    submissions_day = add_title_components(submissions_day, encode, n_components=5).sort_values(
        "day", ascending=False
    )

    subreddits_week = (
        comments_day.groupby(BY_SUBREDDIT_WEEK)[FLAG_COLUMNS].sum()
        .join(united["subreddit"])
        .reset_index()
        .sort_values("week", ascending=False)
    )  # fmt: skip

    flaggers = _at_least(
        flags.groupby(BY_SUBREDDIT_DAY + ["comm_author"])[FLAG_COLUMNS].sum().reset_index(),
        "comm_author", MIN_ROWS,
    ).sort_values("week", ascending=False)  # fmt: skip

    flagged_authors = (
        _at_least(
            flags.groupby(BY_SUBREDDIT_DAY + ["subm_author"])[FLAG_COLUMNS].sum().reset_index(),
            "subm_author", MIN_ROWS,
        )
        .set_index(BY_AUTHOR_WEEK)
        .join(united["author"], how="left")
        .reset_index()
        .sort_values("week", ascending=False)
    )  # fmt: skip

    domains_week = (
        _at_least(
            flags.groupby(BY_DOMAIN_WEEK)[FLAG_COLUMNS].sum().reset_index(),
            "link_domain", MIN_ROWS,
        )
        .set_index(BY_DOMAIN_WEEK)
        .join(united["domain"], how="left")
        .reset_index()
        .sort_values("week", ascending=False)
    )  # fmt: skip

    return {
        "app_flags_day.csv": flags_day,
        "app_subm_day.csv": submissions_day,
        "app_subr_week.csv": subreddits_week,
        "app_authFlaggers_week.csv": flaggers,
        "app_authFlagged_week.csv": flagged_authors,
        "app_domain_week.csv": domains_week,
    }


def run(
    year: int,
    encode: Encoder | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    dashboard_dir: Path | None = None,
    subreddits_file: Path | None = None,
) -> None:
    """Build the UNITED tables and the six dashboard tables for ``year``."""
    raw_dir = raw_dir or settings.RAW_DIR
    processed_dir = processed_dir or settings.PROCESSED_DIR
    dashboard_dir = dashboard_dir or settings.DASHBOARD_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    log.info("Started unification for %d", year)
    encode = encode or load_encoder()
    united = build_united(year, raw_dir, processed_dir, load_tracked_subreddits(subreddits_file))
    for file_name, table in build_dashboard_tables(united, encode).items():
        write_csv(table, dashboard_dir / file_name)
        log.info("Wrote %s. Shape: %s", file_name, table.shape)
