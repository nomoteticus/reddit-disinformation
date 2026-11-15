"""Loading and filtering the tables behind the dashboard.

The tables are produced by ``stroo.aggregate`` (see Appendix A of the paper):

======================  ==============================================
``app_flags_day.csv``   one row per flag sentence
``app_subm_day.csv``    flagged posts, with five title-embedding columns
``app_subr_week.csv``   flags and number of posts per subreddit and week
======================  ==============================================

The dashboard shows no user names. The author tables written by the pipeline
are not read, and user mentions inside comment text and titles are masked.
Posts that were removed by moderators or deleted by their authors are left
out unless ``STROO_SHOW_REMOVED=1`` is set.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import settings

FLAG_TYPE_LABELS = {
    "disinformation": "Dis/Misinfo",
    "fakenews": "Fake news",
    "misleading": "Clickbait",
    "unreliable": "Unreliable",
    "propaganda": "Propaganda",
    "bs": "Bullshit",
    "other": "Other",
}
DEFAULT_FLAG_TYPES = ["disinformation", "fakenews", "misleading", "unreliable", "propaganda"]
COMPONENTS = ["PC01", "PC02", "PC03", "PC04", "PC05"]
N_TOP_SUBREDDITS = 20

USER_MENTION = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{2,}")


def mask_users(text: pd.Series) -> pd.Series:
    """Replace Reddit user mentions (``u/name``) by ``u/[user]``."""
    return text.astype(str).str.replace(USER_MENTION, "u/[user]", regex=True)


def domain_of(url: str) -> str:
    """The web domain of a link, without scheme and ``www.`` (``-`` if not a link)."""
    found = re.search(r"^.+?[a-z]\/", url, re.IGNORECASE)
    if not found:
        return "-"
    domain = re.sub(r"https?\:\/\/", "", found.group(0)[:-1])
    domain = re.sub(r"www\.", "", domain)
    return re.sub("/.*", "", domain)


@dataclass
class DashboardData:
    flags: pd.DataFrame
    submissions: pd.DataFrame
    subreddits: pd.DataFrame
    #: All scraped posts per subreddit, topic and week: denominators for "percent flagged".
    all_posts: pd.DataFrame
    #: First day, last day, number of days and slider label of each week.
    weeks: pd.DataFrame
    top_subreddits: list[str]
    show_removed: bool = False

    @property
    def week_range(self) -> tuple[int, int]:
        return int(self.subreddits["week"].min()), int(self.subreddits["week"].max())

    @property
    def year(self) -> str:
        """Year(s) covered by the data, for labels."""
        years = sorted({day[:4] for day in (self.weeks["min"].min(), self.weeks["max"].max())})
        return "-".join(years)


def load_data(data_dir: Path | None = None, show_removed: bool | None = None) -> DashboardData:
    data_dir = data_dir or settings.DASHBOARD_DATA_DIR
    if show_removed is None:
        show_removed = os.environ.get("STROO_SHOW_REMOVED", "") == "1"
    flags = pd.read_csv(data_dir / "app_flags_day.csv")
    submissions = pd.read_csv(data_dir / "app_subm_day.csv")
    subreddits = pd.read_csv(data_dir / "app_subr_week.csv")

    weeks = submissions.groupby("week")["day"].agg(["min", "max", "nunique"])
    weeks["label"] = weeks["max"].str[5:7] + "/" + weeks["max"].str[8:10]

    if not show_removed:
        flags = flags[flags["subm_removed"] == "visible"]
        submissions = submissions[submissions["subm_id"].isin(set(flags["subm_id"]))]
    flags = flags.assign(sent=mask_users(flags["sent"]), subm_title=mask_users(flags["subm_title"]))
    submissions = submissions.assign(subm_title=mask_users(submissions["subm_title"]))

    subreddits["week"] = subreddits["week"].astype(int)
    subreddits["ISFLAG"] = subreddits["flag"]
    submissions["cluster"] = "1"

    # The linked domain is taken from the first flag of each post.
    flags["domain"] = [domain_of(url) for url in flags["link_url"]]
    first_domain = flags.groupby("subm_id")["domain"].first()
    submissions = submissions.set_index("subm_id").join(first_domain).reset_index()

    top = subreddits.groupby("subreddit")["ISFLAG"].sum().sort_values(ascending=False)
    return DashboardData(
        flags=flags,
        submissions=submissions,
        subreddits=subreddits,
        all_posts=subreddits.groupby(["subreddit", "subm_covid", "week"])[
            ["n_subm_all", "n_subm_kept"]
        ]
        .sum()
        .reset_index(),
        weeks=weeks,
        top_subreddits=list(top.head(N_TOP_SUBREDDITS).index),
        show_removed=show_removed,
    )


# --- Filters (left-hand panel) ------------------------------------------------


@dataclass(frozen=True)
class Filters:
    weeks: tuple[int, int]
    flag_types: tuple[str, ...]
    min_flags: int = 1
    subreddits: tuple[str, ...] = ()
    covid_only: bool = False


def _topic_and_subreddit(table: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    if filters.covid_only:
        table = table[table["subm_covid"]]
    if filters.subreddits:
        table = table[table["subreddit"].isin(filters.subreddits)]
    return table


def _common_filters(
    table: pd.DataFrame, filters: Filters, drop_unflagged: bool = True
) -> pd.DataFrame:
    """Week, flag type, COVID-19 topic and subreddit; shared by every table.

    ``ISFLAG`` is the number of flags of the selected types in each row.
    """
    table = table[table["week"].between(*filters.weeks)].copy()
    table["ISFLAG"] = table[list(filters.flag_types)].sum(axis=1)
    if drop_unflagged:
        table = table[table["ISFLAG"] > 0]
    return _topic_and_subreddit(table, filters)


def filter_flags(data: DashboardData, filters: Filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag sentences and flagged posts that pass the filters.

    A post is kept if it has at least ``min_flags`` flag sentences of the
    selected types.
    """
    flags = _common_filters(data.flags, filters)
    flags["ISFLAG"] = np.sign(flags["ISFLAG"])
    n_flags = flags.groupby("subm_id").agg(nflags=("ISFLAG", "sum"))
    n_flags = n_flags[n_flags["nflags"] >= filters.min_flags]
    submissions = data.submissions.set_index("subm_id").join(n_flags, how="inner").reset_index()
    return flags[flags["subm_id"].isin(submissions["subm_id"])], submissions


def filter_subreddits(data: DashboardData, filters: Filters) -> pd.DataFrame:
    """The subreddit-by-week table for the selection (``min_flags`` does not apply).

    Weeks without flags are kept: they count in the denominator of "percent of
    posts flagged".
    """
    return _common_filters(data.subreddits, filters, drop_unflagged=False)


def to_json(table: pd.DataFrame) -> str:
    return table.to_json(orient="split")


def from_json(text: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(text), orient="split")


# --- Derived tables -----------------------------------------------------------


def flagged_share(posts: pd.DataFrame, data: DashboardData, filters: Filters) -> pd.DataFrame:
    """Flagged posts per week: number, per day, and as a percentage of all posts.

    ``posts`` are the flagged posts that pass the filters. The percentage is
    taken over all scraped posts of the same subreddits and topic.
    """
    all_posts = _topic_and_subreddit(data.all_posts, filters)
    all_posts = all_posts.groupby("week")[["n_subm_all", "n_subm_kept"]].sum()
    table = (
        posts.groupby("week").agg(n_flags=("subm_id", "count"))
        .join(all_posts)
        .join(data.weeks["nunique"])
        .reset_index()
    )  # fmt: skip
    table["n_day"] = (table["n_flags"] / table["nunique"]).round().astype(int)
    table["perc_FLAG"] = table["n_flags"] / table["n_subm_kept"]
    table["perc_FLAG_pr"] = (100 * table["perc_FLAG"]).round(1)
    return table


def flag_rate(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Flags and share of flagged posts per group, for subreddit-level tables."""
    table = table.groupby(by)[["n_subm_kept", "ISFLAG"]].sum().sort_index().reset_index()
    return table.assign(perc_FLAG=table["ISFLAG"] / table["n_subm_kept"])


def flagged_domains(flags: pd.DataFrame) -> pd.DataFrame:
    """Linked websites ranked by the number of flagged posts in the selection."""
    table = flags.groupby("domain").agg(
        flagged_posts=("subm_id", "nunique"), flags=("sent", "count")
    )
    table = table.sort_values(["flagged_posts", "flags"], ascending=False).reset_index()
    return table.rename(
        columns={"domain": "website", "flagged_posts": "flagged posts", "flags": "flag sentences"}
    )


def similar_posts(
    data: DashboardData, subm_id: str, threshold: float = 0.005, max_posts: int = 20
) -> pd.DataFrame:
    """Flagged posts whose titles are close to this one in embedding space.

    Distances are computed for this post only, so no post-by-post distance
    matrix has to be kept in memory.
    """
    posts = data.submissions
    this = posts.loc[posts["subm_id"] == subm_id, COMPONENTS].to_numpy()[0]
    distance = np.sqrt(((posts[COMPONENTS].to_numpy() - this) ** 2).sum(axis=1))
    nearest = pd.Series(distance, index=posts.index).sort_values(kind="stable").head(max_posts)
    similar = posts.loc[nearest[nearest < threshold].index, ["day", "subm_title", "subreddit",
                                                             "domain"]]  # fmt: skip
    return similar.assign(subm_title=shorten(similar["subm_title"], 60)).sort_values("day")


def shorten(titles, n_characters: int) -> list[str]:
    return [title[:n_characters] + "..." for title in titles]
