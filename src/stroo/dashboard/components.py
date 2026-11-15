"""Reusable pieces of the page: flex containers, tables, post cards, word lists."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from dash import html
from spacy.lang.en.stop_words import STOP_WORDS

from .data import DashboardData, shorten, similar_posts

IGNORED_WORDS = STOP_WORDS | {"new", "says"}


# --- Flex containers (classes defined in assets/styles.css) ---------------------


def _container(children, class_name, size=None, extra=None, shadow=False, stretch=True):
    classes = [class_name]
    if not stretch:
        classes.append("nostretch")
    if shadow:
        classes.append("shadow_container")
    if size:
        classes.append(f"flx{size}")
    if extra:
        classes.append(extra)
    return html.Div(children=children, className=" ".join(classes))


def row(children, **kwargs):
    """Children side by side. ``size`` (1-12) is the relative width."""
    return _container(children, "cont_generic", **kwargs)


def column(children, **kwargs):
    """Children stacked vertically."""
    return _container(children, "cont_generic v", **kwargs)


def item(children, **kwargs):
    return _container(children, "panel", **kwargs)


def paragraphs(texts) -> html.Div:
    return html.Div([html.P(text) for text in texts])


def bullets(texts) -> html.Ul:
    return html.Ul([html.Li(text) for text in texts])


def badge(text: str, class_name: str = "badge badge-warning ml-1") -> html.Span:
    return html.Span(text, className=class_name)


def info_box(header: str, content_id: str, number_class: str = "bigger_number") -> html.Div:
    """A framed headline number, filled in by a callback."""
    return html.Div(
        [html.P(header), html.Div(id=content_id, className=number_class)],
        className="shadow_container info_box",
    )


# --- Tables -------------------------------------------------------------------


def table(frame: pd.DataFrame, max_rows: int = 10, class_name: str = "normaltable",
          link_column: str | None = None, link_prefix: str = "") -> html.Table:  # fmt: skip
    """An HTML table of the first ``max_rows`` rows; one column can hold links."""

    def cell(value, column_name):
        if column_name == link_column:
            return html.Td(html.A(value, href=link_prefix + str(value)))
        return html.Td(value)

    header = html.Tr([html.Th(name) for name in frame.columns])
    body = [
        html.Tr([cell(value, name) for name, value in zip(frame.columns, values, strict=True)])
        for values in frame.head(max_rows).itertuples(index=False)
    ]
    return html.Table([header, *body], className=class_name)


# --- Words --------------------------------------------------------------------


def _words(texts) -> list[str]:
    """Lower-cased words of the titles, without stop words and pure numbers."""
    return [
        word.lower()
        for word in " ".join(texts).split()
        if word.lower() not in IGNORED_WORDS and re.search("[a-z]", word)
    ]


def common_words(texts, n_words: int = 100) -> pd.DataFrame:
    """The most frequent words, with their share of all words in percent."""
    share = pd.Series(_words(texts)).value_counts(normalize=True).head(n_words)
    return pd.DataFrame({"word": share.index, "freq": (100 * share.to_numpy()).round(2)})


def cluster_words(posts: pd.DataFrame, cluster: int, n_words: int = 5):
    """A line of badges with the most frequent title words of one cluster."""
    titles = posts.loc[posts["cluster"] == str(cluster), "subm_title"]
    words = [word for word, _ in Counter(_words(titles)).most_common(n_words)]
    return row(
        [
            item(html.Strong(f"CL{cluster}: "), size=1),
            item(html.Span([badge(word, "badge badge-yellow mr-1") for word in words]), size=10),
        ],
        extra="cluster_sep",
    )


# --- Flagged posts --------------------------------------------------------------


def _post_header(post) -> html.Div:
    return html.Div(
        [
            html.A("r/" + post.subreddit, href=post.subm_link,
                   className="subcontainer_subm subreddit"),
            html.Span(post.day, className="subcontainer_subm date"),
            html.A(post.domain, href=post.link_url, className="subcontainer_subm domain"),
        ],
        className="subcontainer_subm",
    )  # fmt: skip


def _flag_list(flags: pd.DataFrame) -> html.Div:
    sentences = [html.P("• " + sentence, className="cont_subm_flags right item")
                 for sentence in flags["sent"]]  # fmt: skip
    return html.Div(
        [
            html.Div("Flags: ", className="cont_subm_flags left"),
            html.Div(html.Div(sentences), className="cont_subm_flags right"),
        ],
        className="cont_subm_flags",
    )


def post_card(flags: pd.DataFrame, class_name: str) -> html.Div:
    """A flagged post with the sentences that flag it (Highlights tab)."""
    post = flags.iloc[0]
    return html.Div(
        [_post_header(post), html.H6(post.subm_title, className="subm_title"), _flag_list(flags)],
        className=class_name,
    )


def post_card_with_analysis(flags: pd.DataFrame, data: DashboardData, class_name: str):
    """A flagged post that unfolds to show its flags and similar posts (Posts tab)."""
    post = flags.iloc[0]
    similar = similar_posts(data, post.subm_id)
    return html.Div(
        [
            _post_header(post),
            html.Details(
                [
                    html.Summary(
                        row(
                            [
                                item(html.Div(post.subm_title,
                                              className="subm_title bold smallpadding truncate"),
                                     size=11),
                                item(html.Span("▼Analyze▼",
                                               className="btn btn-danger mr-1 smallpadding")),
                            ],
                            extra="cont_subm_title",
                        )
                    ),
                    html.Div("Deletion status: " + str(post.subm_removed),
                             className="cont_subm_flags left"),
                    _flag_list(flags),
                    html.Div("Similar posts on Reddit flagged as false information:",
                             className="subm_title"),
                    table(similar, max_rows=len(similar), class_name="small"),
                ],
                className="post_details",
            ),
        ],
        className=class_name,
    )  # fmt: skip


def latest_posts(flags: pd.DataFrame, n_posts: int) -> list[pd.DataFrame]:
    """The flag sentences of the ``n_posts`` most recently flagged posts, one table each."""
    newest = flags["subm_id"].drop_duplicates().head(n_posts)
    return [flags[flags["subm_id"] == subm_id] for subm_id in newest]


__all__ = [
    "badge", "bullets", "cluster_words", "column", "common_words", "info_box", "item",
    "latest_posts", "paragraphs", "post_card", "post_card_with_analysis", "row", "shorten",
    "table",
]  # fmt: skip
