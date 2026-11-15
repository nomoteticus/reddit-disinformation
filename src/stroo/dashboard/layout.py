"""Page layout: header with the week slider, filter panel, and one function per tab."""

from __future__ import annotations

from dash import dcc, html

from . import text
from .components import bullets, column, info_box, item, paragraphs, row
from .data import DEFAULT_FLAG_TYPES, FLAG_TYPE_LABELS, DashboardData

WEEKS_SHOWN = 4
DEFAULT_MIN_FLAGS = 2
DEFAULT_N_CLUSTERS = 3  # Section 7.3 of the paper
LIST_SIZES = [10, 20, 50, 100]

TABS = {
    "highlights": "Highlights",
    "flags": "Flags",
    "posts": "Submissions",
    "subreddits": "Subreddits",
    "clustering": "Clustering",
    "domains": "Domains",
    "authors": "Authors",
    "method": "Method",
}


def _radio(component_id: str, options, value, labels=None):
    labels = labels or [str(option) for option in options]
    return dcc.RadioItems(
        id=component_id,
        options=[{"label": label, "value": option}
                 for option, label in zip(options, labels, strict=True)],
        value=value,
        labelStyle={"display": "inline-block"},
    )  # fmt: skip


# --- Header and filter panel ----------------------------------------------------


def week_slider(data: DashboardData) -> dcc.RangeSlider:
    first, last = data.week_range
    step = 1 + (last - first) // 14
    marked = sorted({*range(first, last, step), last})
    return dcc.RangeSlider(
        id="sl_week",
        min=first,
        max=last,
        step=1,
        value=[max(last - WEEKS_SHOWN, first), last],
        marks={week: {"label": data.weeks["label"].get(week, "")} for week in marked},
        className="time_slider",
    )


def filter_panel(data: DashboardData, version: str) -> html.Div:
    return html.Div(
        className="item south_gradient west",
        children=[
            html.Div(html.I("version " + version)),
            html.H5("⬐ Filter data", className="filter_data"),
            row(column([
                html.H5("Flag count"),
                dcc.Dropdown(
                    id="drop_min_nflags",
                    value=DEFAULT_MIN_FLAGS,
                    options=[{"label": f"minimum {n}", "value": n} for n in [1, 2, 3, 5, 10]],
                    multi=False, clearable=False, searchable=False,
                    className="left_dropdowns",
                ),
                html.H5("Type of flag"),
                dcc.Checklist(
                    id="cl_flagtypes",
                    value=DEFAULT_FLAG_TYPES,
                    options=[{"label": label, "value": flag_type}
                             for flag_type, label in FLAG_TYPE_LABELS.items()],
                    className="checklist_labels",
                ),
                html.H5("Subreddits"),
                dcc.Dropdown(
                    id="dr_subr20_subm",
                    options=[{"label": name.capitalize(), "value": name}
                             for name in sorted(data.top_subreddits)],
                    multi=True, clearable=False,
                    placeholder="Filter by subreddit...",
                    className="left_dropdowns",
                ),
                html.H5("Coronavirus only?"),
                dcc.RadioItems(
                    id="rd_covid",
                    value="no",
                    options=[{"label": "yes", "value": "yes"}, {"label": "no", "value": "no"}],
                    className="checklist_labels",
                ),
            ])),
        ],
    )  # fmt: skip


def page(data: DashboardData, logo_url: str, version: str) -> html.Div:
    return html.Div(
        className="page",
        children=[
            # Filtered tables, shared by the callbacks of all tabs.
            dcc.Store(id="flag_filt_json"),
            dcc.Store(id="subm_filt_json"),
            dcc.Store(id="filters"),
            html.Div(
                className="header",
                children=[
                    html.Div(
                        className="item north west",
                        children=[
                            html.Img(src=logo_url, className="nw_stroo"),
                            html.Span(html.H5(html.B("Is it true?")), className="nw_stroo_txt"),
                        ],
                    ),
                    html.Div(
                        className="item north east",
                        children=[
                            html.Div(
                                [
                                    html.Span("Monitoring perceived disinformation on Reddit ",
                                              className="dates_banner"),
                                    html.Span(id="show_dates", className="dates_banner_bold"),
                                ],
                                className="title_ne",
                            ),
                            html.Div(week_slider(data), className="title_ne"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="layout",
                children=[
                    filter_panel(data, version),
                    html.Div(
                        className="item south east content",
                        children=html.Div([
                            dcc.Tabs(
                                id="tabs",
                                value="highlights",
                                parent_className="custom-tabs",
                                className="custom-tabs-container",
                                children=[
                                    dcc.Tab(label=label, value=value, className="custom-tab",
                                            selected_className="custom-tab--selected")
                                    for value, label in TABS.items()
                                ],
                            ),
                            html.Div(id="tab-content"),
                        ]),
                    ),
                ],
            ),
        ],
    )  # fmt: skip


# --- Tabs -----------------------------------------------------------------------


def highlights_tab() -> html.Div:
    return html.Div([
        html.Br(),
        paragraphs(text.HIGHLIGHTS),
        html.H5("Posts flagged by Reddit users as false information"),
        row([
            row([
                info_box("# Flagged posts:", "highl_n_subm"),
                info_box("# Subreddits:", "highl_n_subr"),
                info_box("# Flagged websites:", "highl_n_wdom"),
                info_box("#1 Keyword:", "highl_top_keyw"),
                info_box("Top informal flags (comments)", "highl_top_flag", "big_number"),
            ], stretch=True, size=7),
            row(column(item([
                row(dcc.Graph(id="plot_evol_line_n")),
                row(_radio("rd_evo_week", ["n_day", "n_flags", "perc_FLAG_pr"], "n_day",
                           ["Daily (N/day)", "Total (N)", "Percent (%)"])),
            ], shadow=True)), size=5),
        ]),
        html.H5("Most recent flagged posts"),
        html.Div(id="highl_subm_detail"),
        html.H5("Common keywords in titles of flagged posts"),
        row([
            item(dcc.Graph(id="plot_highl_word_count"), shadow=True),
            item(dcc.Graph(id="plot_two_keywords"), shadow=True),
        ]),
    ])  # fmt: skip


def flags_tab() -> html.Div:
    return html.Div([
        html.Br(),
        paragraphs(text.FLAGS),
        row([item("Number of results to display:", size=3),
             item(_radio("rd_ntopflags", LIST_SIZES, 10), size=4)]),
        html.H5("Most common flags"),
        html.Div(id="tbl_flags_freq"),
        html.H5("Most recent flags"),
        html.Div(id="tbl_flags_recent"),
    ])  # fmt: skip


def posts_tab() -> html.Div:
    return html.Div([
        html.Br(),
        paragraphs(text.SUBMISSIONS),
        row([item("Number of submissions to display:", size=3),
             item(_radio("rd_ntopsubs", LIST_SIZES, 10), size=4)]),
        html.Br(),
        html.Div(id="subm_subm_detail"),
    ])  # fmt: skip


def subreddits_tab() -> html.Div:
    statistic = [["n", "pct"], "n", ["Frequency (N)", "Percent of subm. (%)"]]
    return html.Div([
        html.Br(),
        html.P(text.AGGREGATE_NOTE),
        html.H5("Subreddits with most flagged submissions"),
        dcc.Graph(id="plot_subr_treemap"),
        html.H5("Evolution of flagged submissions"),
        row([
            column([html.H6("Per subreddit category"),
                    _radio("rd_subr_all_stat", *statistic),
                    dcc.Graph(id="plot_subr_line_cat")], size=6),
            column([html.H6("Per subreddit (20 most flagged)"),
                    _radio("rd_subr_20_stat", *statistic),
                    dcc.Graph(id="plot_subr_line_20")], size=6),
        ]),
    ])  # fmt: skip


def cluster_options(n_clusters: int) -> list[dict]:
    return [{"label": f"Cluster #{n}", "value": n} for n in range(1, n_clusters + 1)]


def clustering_tab() -> html.Div:
    return html.Div([
        html.Div([
            html.Br(),
            html.P("Topics are computed based on the submission titles "
                   "(Universal Sentence Embeddings + PCA + K-means)"),
        ]),
        row([
            item(column([
                item(html.H5("1. Choose number of clusters")),
                item(html.Div(dcc.Slider(
                    id="sl_nclust", min=1, max=10, step=1, value=DEFAULT_N_CLUSTERS,
                    marks={n: str(n) for n in range(1, 11)},
                ))),
                item(html.H6("Top keywords / cluster")),
                item(html.Div(id="badges_topwords")),
            ]), size=5),
            column([
                item(dcc.Graph(id="plot_kmeans"), shadow=True),
                row([
                    item("Choose type of plot:", size=2),
                    item(_radio("rd_clust_which", ["evolution", "differentiation"],
                                "evolution"), size=3),
                ]),
            ], size=6),
        ]),
        html.Br(),
        row([
            item(html.H5("2. Choose cluster for inspection:"), size=5),
            item(dcc.Dropdown(id="dr_clustersubm", value=1, multi=False, clearable=False,
                              options=cluster_options(DEFAULT_N_CLUSTERS),
                              placeholder="Choose cluster"), size=2),
            item("", size=4),
        ]),
        html.H6("The most recent titles in the cluster:"),
        html.Div(id="tbl_clustersubm"),
    ])  # fmt: skip


def _ranking_tab(title: str, prefix: str, min_cases: list[int]) -> html.Div:
    """Shared layout of the Domains and Authors tabs."""
    return html.Div([
        html.Br(),
        html.P(text.AGGREGATE_NOTE),
        html.H5(title),
        row([
            column([html.Div("Display maximum rows:"),
                    _radio(f"rd_ntop{prefix}", [10, 20, 50], 10)], size=3),
            column([html.Div("Sort by:"),
                    _radio(f"rd_{prefix}percent", ["frequency", "percent"], "percent")], size=3),
            column([html.Div("Minimum cases for percentage:",
                             title="Only show percentages if the number of submissions is at "
                                   "least this number."),
                    _radio(f"rd_mincases{prefix}", min_cases, 10)], size=5),
        ]),
        html.Br(),
        html.Div(id=f"tbl_{prefix}"),
    ])  # fmt: skip


def domains_tab() -> html.Div:
    return _ranking_tab("Most flagged domains", "domains", [10, 20, 50])


def authors_tab() -> html.Div:
    return _ranking_tab("Most flagged submission authors", "authors", [10, 20, 50, 100])


def method_tab(pipeline_image: str, patterns_image: str, period: str) -> html.Div:
    arrows = ["➝"] * (len(text.PIPELINE_STEPS) - 1) + [""]
    collection = [line.format(period=period) for line in text.METHOD_COLLECTION]
    return html.Div([
        html.Br(),
        paragraphs(text.METHOD_INTRO),
        html.Br(),
        html.H5("1. Data collection"),
        paragraphs(collection),
        bullets(text.SUBREDDIT_CRITERIA),
        paragraphs(text.METHOD_COLLECTION_END),
        html.Br(),
        html.H5("2. Detecting informal flags"),
        paragraphs(text.METHOD_FLAGS),
        html.Div([
            html.Span([html.Span(step, className="badge badge-warning ml-1"), arrow])
            for step, arrow in zip(text.PIPELINE_STEPS, arrows, strict=True)
        ]),
        html.Br(),
        row([
            column([
                html.H6("2a. Keyword filtering"),
                html.P(text.METHOD_KEYWORDS_INTRO),
                bullets(text.METHOD_KEYWORDS),
                html.H6("2b. Pattern matching"),
            ], size=5),
            column(html.Img(src=pipeline_image, className="method_pipeline"), size=7),
        ]),
        paragraphs(text.METHOD_PATTERNS),
        html.Div(html.Img(src=patterns_image, className="method_patterns")),
        paragraphs(text.METHOD_PATTERNS_END),
        html.H5("3. Clustering submissions"),
        paragraphs(text.METHOD_CLUSTERING),
        html.Br(),
        paragraphs(text.METHOD_END),
        html.P(["Read more: ", html.A(text.PAPER_CITATION, href=text.PAPER_URL)]),
    ])  # fmt: skip
