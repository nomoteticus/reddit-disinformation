"""Callbacks: one filter step feeding every tab, then one or two callbacks per tab."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from dash import Dash, Input, Output, html
from sklearn.cluster import MiniBatchKMeans

from . import components, figures, layout
from .data import (
    COMPONENTS,
    DashboardData,
    Filters,
    filter_aggregate,
    filter_flags,
    flag_rate,
    flagged_share,
    from_json,
    most_flagged,
    shorten,
    to_json,
)

CARD_CLASS = "shadow_container container_subm"
KEYWORDS_COMPARED = ("trump", "biden")
CLUSTER_SEED = 777


def _pretty_date(day: str, with_year: bool) -> str:
    return datetime.strptime(day, "%Y-%m-%d").strftime("%b %d, %Y" if with_year else "%b %d")


def _filters(stored: dict) -> Filters:
    return Filters(
        weeks=tuple(stored["weeks"]),
        flag_types=tuple(stored["flag_types"]),
        min_flags=stored["min_flags"],
        subreddits=tuple(stored["subreddits"]),
        covid_only=stored["covid_only"],
    )


def register_callbacks(app: Dash, data: DashboardData) -> None:  # noqa: C901 - a flat list
    weeks = data.weeks

    # --- Header and filters -------------------------------------------------------

    @app.callback(Output("show_dates", "children"), Input("sl_week", "value"))
    def show_dates(week_range):
        first = _pretty_date(weeks.loc[week_range[0], "min"], with_year=False)
        last = _pretty_date(weeks.loc[week_range[1], "max"], with_year=True)
        return f"⬐ from {first} to {last} ⬎"

    @app.callback(
        Output("flag_filt_json", "data"),
        Output("subm_filt_json", "data"),
        Output("filters", "data"),
        Input("sl_week", "value"),
        Input("cl_flagtypes", "value"),
        Input("drop_min_nflags", "value"),
        Input("dr_subr20_subm", "value"),
        Input("rd_covid", "value"),
    )
    def apply_filters(week_range, flag_types, min_flags, subreddits, covid):
        stored = {
            "weeks": week_range,
            "flag_types": flag_types,
            "min_flags": min_flags,
            "subreddits": subreddits or [],
            "covid_only": covid == "yes",
        }
        flags, posts = filter_flags(data, _filters(stored))
        return to_json(flags), to_json(posts), stored

    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        if tab == "method":
            first, last = weeks["min"].min(), weeks["max"].max()
            period = f"{_pretty_date(first, True)} to {_pretty_date(last, True)}"
            return layout.method_tab(
                app.get_asset_url("method_pipeline.png"),
                app.get_asset_url("method_patterns.png"),
                period,
            )
        return getattr(layout, f"{tab}_tab")()

    # --- Highlights ---------------------------------------------------------------

    @app.callback(
        Output("highl_n_subm", "children"),
        Output("highl_n_subr", "children"),
        Output("highl_n_wdom", "children"),
        Output("highl_top_keyw", "children"),
        Output("highl_top_flag", "children"),
        Input("flag_filt_json", "data"),
        Input("subm_filt_json", "data"),
    )
    def headline_numbers(flags_json, posts_json):
        flags, posts = from_json(flags_json), from_json(posts_json)
        if posts.empty:
            return "0", "0", "0", "-", "-"
        words = components.common_words(posts["subm_title"])
        top_flags = flags["sent"].str.replace("[.,;]$", "", regex=True).value_counts().head(2)
        return (
            str(len(posts)),
            str(posts["subreddit"].nunique()),
            str(posts["domain"].nunique()),
            words["word"].iloc[0].capitalize() if len(words) else "-",
            " | ".join(top_flags.index),
        )

    @app.callback(
        Output("plot_evol_line_n", "figure"),
        Input("subm_filt_json", "data"),
        Input("rd_evo_week", "value"),
    )
    def flagged_over_time(posts_json, statistic):
        by_week = flagged_share(from_json(posts_json), data, by=["week"])
        return figures.flagged_over_time(by_week, weeks, statistic, data.year)

    @app.callback(Output("highl_subm_detail", "children"), Input("flag_filt_json", "data"))
    def latest_flagged_posts(flags_json):
        posts = components.latest_posts(from_json(flags_json), n_posts=5)
        cards = [components.post_card(post, CARD_CLASS) for post in posts]
        if len(cards) <= 1:
            return cards
        more = html.Details([html.Summary("Show / Hide more submissions",
                                          className="btn btn-light mb-3"), *cards[1:]])  # fmt: skip
        return [cards[0], more]

    @app.callback(Output("plot_highl_word_count", "figure"), Input("subm_filt_json", "data"))
    def keyword_frequencies(posts_json):
        posts = from_json(posts_json)
        words = components.common_words(posts["subm_title"], n_words=10) if len(posts) else posts
        return figures.word_frequencies(words.sort_values("freq") if len(words) else words)

    @app.callback(Output("plot_two_keywords", "figure"), Input("subm_filt_json", "data"))
    def two_keywords_over_time(posts_json):
        posts = from_json(posts_json)
        if posts.empty:
            return figures.two_keywords(posts, weeks)
        by_week = pd.concat(
            components.common_words(titles).assign(week=week)
            for week, titles in posts.groupby("week")["subm_title"]
        )
        by_week = by_week[by_week["word"].isin(KEYWORDS_COMPARED)]
        return figures.two_keywords(by_week, weeks)

    # --- Flags ----------------------------------------------------------------------

    @app.callback(
        Output("tbl_flags_freq", "children"),
        Output("tbl_flags_recent", "children"),
        Input("flag_filt_json", "data"),
        Input("rd_ntopflags", "value"),
    )
    def flag_tables(flags_json, n_rows):
        flags = from_json(flags_json)
        common = flags["sent"].value_counts().head(n_rows).reset_index()
        common.columns = ["flag", "count"]
        return (
            components.table(common, max_rows=n_rows),
            components.table(flags[["subreddit", "sent", "day"]], max_rows=n_rows),
        )

    # --- Submissions ----------------------------------------------------------------

    @app.callback(
        Output("subm_subm_detail", "children"),
        Input("flag_filt_json", "data"),
        Input("rd_ntopsubs", "value"),
    )
    def flagged_post_list(flags_json, n_posts):
        posts = components.latest_posts(from_json(flags_json), n_posts)
        return [components.post_card_with_analysis(post, data, CARD_CLASS) for post in posts]

    # --- Subreddits -------------------------------------------------------------------

    def _subreddit_weeks(stored) -> pd.DataFrame:
        table = filter_aggregate(data.subreddits, _filters(stored), drop_unflagged=False)
        return flag_rate(table, ["subreddit_cat", "subreddit", "week"])

    @app.callback(Output("plot_subr_treemap", "figure"), Input("filters", "data"))
    def subreddit_treemap(stored):
        by_subreddit = flag_rate(_subreddit_weeks(stored), ["subreddit_cat", "subreddit"])
        return figures.subreddit_treemap(by_subreddit)

    @app.callback(
        Output("plot_subr_line_cat", "figure"),
        Input("filters", "data"),
        Input("rd_subr_all_stat", "value"),
    )
    def flags_by_category(stored, statistic):
        by_week = flag_rate(_subreddit_weeks(stored), ["subreddit_cat", "week"])
        return figures.flags_by_group(by_week, "subreddit_cat", statistic, weeks)

    @app.callback(
        Output("plot_subr_line_20", "figure"),
        Input("filters", "data"),
        Input("rd_subr_20_stat", "value"),
    )
    def flags_by_subreddit(stored, statistic):
        table = _subreddit_weeks(stored)
        table = table[table["subreddit"].isin(data.top_subreddits)]
        by_week = flag_rate(table, ["subreddit", "week"])
        return figures.flags_by_group(by_week, "subreddit", statistic, weeks)

    # --- Clustering -------------------------------------------------------------------

    @app.callback(Output("dr_clustersubm", "options"), Input("sl_nclust", "value"))
    def cluster_choices(n_clusters):
        return layout.cluster_options(n_clusters)

    @app.callback(
        Output("plot_kmeans", "figure"),
        Output("badges_topwords", "children"),
        Output("tbl_clustersubm", "children"),
        Input("subm_filt_json", "data"),
        Input("sl_nclust", "value"),
        Input("dr_clustersubm", "value"),
        Input("rd_clust_which", "value"),
    )
    def cluster_posts(posts_json, n_clusters, selected, which_plot):
        posts = from_json(posts_json)
        if len(posts) < n_clusters:
            empty = figures.clusters(posts.iloc[:0])
            return empty, "Not enough posts for this number of clusters.", None
        if n_clusters > 1:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=1000, random_state=CLUSTER_SEED
            ).fit(posts[COMPONENTS])
            posts["cluster"] = (kmeans.predict(posts[COMPONENTS]) + 1).astype(str)
            posts = posts.reset_index().sort_values("cluster")
        posts["cluster"] = posts["cluster"].astype(str)
        posts["week"] = posts["week"].astype(int)

        if which_plot == "evolution":
            counts = posts.groupby(["week", "cluster"])["flag"].count()
            share = (100 * counts / counts.groupby(level="week").transform("sum")).reset_index()
            figure = figures.clusters_over_time(share.sort_values(["cluster", "week"]), weeks)
        else:
            figure = figures.clusters(posts)

        words = [components.cluster_words(posts, n, n_words=5) for n in range(1, n_clusters + 1)]

        in_cluster = posts.loc[
            posts["cluster"] == str(selected or 1), ["day", "subm_title", "subreddit", "domain"]
        ].sort_values("day", ascending=False).head(100)  # fmt: skip
        in_cluster = in_cluster.assign(
            subm_title=shorten(in_cluster["subm_title"], 75), day=in_cluster["day"].str[5:]
        )
        return figure, words, components.table(in_cluster, max_rows=30, class_name="small")

    # --- Domains and authors ------------------------------------------------------------

    @app.callback(
        Output("tbl_domains", "children"),
        Input("filters", "data"),
        Input("rd_ntopdomains", "value"),
        Input("rd_mincasesdomains", "value"),
        Input("rd_domainspercent", "value"),
    )
    def domain_table(stored, n_rows, min_posts, sort_by):
        table = filter_aggregate(data.domains, _filters(stored))
        return components.table(
            most_flagged(table, "link_domain", min_posts, sort_by), max_rows=n_rows
        )

    @app.callback(
        Output("tbl_authors", "children"),
        Input("filters", "data"),
        Input("rd_ntopauthors", "value"),
        Input("rd_mincasesauthors", "value"),
        Input("rd_authorspercent", "value"),
    )
    def author_table(stored, n_rows, min_posts, sort_by):
        table = filter_aggregate(data.flagged_authors, _filters(stored))
        return components.table(
            most_flagged(table, "subm_author", min_posts, sort_by),
            max_rows=n_rows,
            link_column="subm_author",
            link_prefix="https://www.reddit.com/user/",
        )
