"""Plotly figures of the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import shorten

BACKGROUND = "LightGoldenRodYellow"
MAIN_COLOR = "brown"

EVOLUTION_Y_TITLES = {
    "n_day": "Number of flagged posts / day",
    "n_flags": "Number of flagged posts / week",
    "perc_FLAG_pr": "Percent of flagged posts (%)",
}


def _style(fig: go.Figure, single_color: bool = True) -> go.Figure:
    """The dashboard's look: yellow background, small margins, brown marks."""
    if single_color:
        fig.update_traces(marker_color=MAIN_COLOR)
        fig.update_traces(line_color=MAIN_COLOR, selector={"type": "scatter"})
    fig.update_xaxes(linecolor="black", gridwidth=0)
    fig.update_layout(
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def _empty(message: str = "No data for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, single_color=False)


def _week_ticks(fig: go.Figure, weeks: pd.DataFrame, n_ticks: int = 5) -> dict:
    """Label the week axis with the last day (month/day) of about ``n_ticks`` weeks."""
    first = int(min(min(trace.x) for trace in fig.data))
    last = int(max(max(trace.x) for trace in fig.data))
    ticks = list(range(first, last, (last - first) // n_ticks + 1)) + [last]
    return {
        "tickmode": "array",
        "tickvals": ticks,
        "ticktext": weeks["label"].reindex(ticks).fillna("").tolist(),
    }


def flagged_over_time(table: pd.DataFrame, weeks: pd.DataFrame, statistic: str, year: str):
    """Flagged posts per week: daily average, weekly total or percent of all posts."""
    if table.empty:
        return _empty()
    fig = px.line(table, x="week", y=statistic, width=300, height=250)
    fig.update_traces(mode="markers+lines")
    fig.update_yaxes(rangemode="tozero", linecolor="black", gridwidth=1)
    fig.update_layout(
        xaxis=_week_ticks(fig, weeks),
        xaxis_title=f"Month/Day of {year}",
        yaxis_title=EVOLUTION_Y_TITLES[statistic],
    )
    return _style(fig)


def word_frequencies(words: pd.DataFrame) -> go.Figure:
    if words.empty:
        return _empty()
    fig = px.bar(words, y="word", x="freq", height=300, width=300, orientation="h")
    fig.update_layout(xaxis_title="Word Frequency (%)", yaxis_title="Keyword")
    return _style(fig)


def two_keywords(words_by_week: pd.DataFrame, weeks: pd.DataFrame) -> go.Figure:
    """Weekly frequency of two keywords in the titles of flagged posts."""
    if words_by_week.empty:
        return _empty("Neither keyword occurs in this selection")
    fig = px.line(words_by_week, x="week", y="freq", color="word", height=300, width=300)
    fig.update_layout(
        xaxis=_week_ticks(fig, weeks), xaxis_title="Week ending", yaxis_title="Word frequency (%)"
    )
    return _style(fig, single_color=False)


def clusters(posts: pd.DataFrame) -> go.Figure:
    """Flagged posts on the first two principal components, coloured by cluster."""
    if posts.empty:
        return _empty()
    posts = posts.assign(Title=shorten(posts["subm_title"], 50))
    fig = px.scatter(
        posts, x="PC01", y="PC02", color="cluster", opacity=0.75,
        hover_data={"cluster": True, "Title": True, "PC01": False, "PC02": False},
    )  # fmt: skip
    fig.update_traces(marker={"size": 5})
    return _style(fig, single_color=False)


def clusters_over_time(share_by_week: pd.DataFrame, weeks: pd.DataFrame) -> go.Figure:
    """Share of each cluster among the flagged posts of each week."""
    if share_by_week.empty:
        return _empty()
    fig = px.line(share_by_week, x="week", y="flag", color="cluster")
    fig.update_layout(
        xaxis=_week_ticks(fig, weeks),
        xaxis_title="Week ending",
        yaxis_title="% of flagged submissions in cluster",
    )
    return _style(fig, single_color=False)


def subreddit_treemap(by_subreddit: pd.DataFrame) -> go.Figure:
    """Subreddits sized by number of flags, grouped by category (Figure 5)."""
    by_subreddit = by_subreddit[by_subreddit["ISFLAG"] > 0]
    if by_subreddit.empty:
        return _empty()
    fig = px.treemap(
        by_subreddit, path=["subreddit_cat", "subreddit"], values="ISFLAG", color="subreddit_cat"
    )
    return _style(fig, single_color=False)


def flags_by_group(by_week: pd.DataFrame, group: str, statistic: str, weeks: pd.DataFrame):
    """Weekly flags per subreddit category or per subreddit: counts or percent (Figure 2)."""
    if by_week.empty:
        return _empty()
    fig = px.line(by_week, x="week", y="ISFLAG" if statistic == "n" else "perc_FLAG", color=group)
    fig.update_traces(mode="markers+lines")
    fig.update_layout(
        xaxis=_week_ticks(fig, weeks),
        xaxis_title="Week ending",
        yaxis_title="Flagged posts" if statistic == "n" else "Share of posts flagged",
    )
    if statistic != "n":
        fig.update_yaxes(tickformat=".0%")
    return _style(fig, single_color=False)
