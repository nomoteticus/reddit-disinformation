"""Dashboard data layer on a small hand-made dataset, plus smoke tests of the app."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from stroo.dashboard import data as dd

TYPES = ["disinformation", "fakenews", "bs", "misleading", "unreliable", "propaganda", "other"]


def _flag_row(subm_id, subreddit, week, day, sent, flag_type, covid=False):
    row = {
        "subreddit_cat": "generic_news", "subreddit": subreddit, "subm_id": subm_id,
        "subm_covid": covid, "subm_title": f"Title of {subm_id}",
        "subm_link": f"https://www.reddit.com/{subm_id}",
        "link_url": f"https://www.site-{subm_id}.com/story", "subm_removed": "visible",
        "sent_id": 0, "sent": sent, "month": 7, "week": week, "day": day, "flag": 1,
    }  # fmt: skip
    row.update({t: int(t == flag_type) for t in TYPES})
    return row


@pytest.fixture
def tiny(tmp_path):
    """Three flagged posts in two subreddits over two weeks."""
    flags = pd.DataFrame([
        _flag_row("a", "news", 28, "2020-07-07", "fake news.", "fakenews", covid=True),
        _flag_row("a", "news", 28, "2020-07-07", "this is propaganda", "propaganda", covid=True),
        _flag_row("b", "worldnews", 29, "2020-07-14", "bullshit", "bs"),
        _flag_row("c", "worldnews", 29, "2020-07-13", "misleading title", "misleading"),
    ])  # fmt: skip
    keys = ["subreddit_cat", "subreddit", "subm_covid", "month", "week", "day", "subm_id",
            "subm_title"]  # fmt: skip
    posts = flags.groupby(keys)[["flag", *TYPES]].sum().reset_index()
    for i, name in enumerate(dd.COMPONENTS):
        posts[name] = [0.001 * (i + 1), 0.001 * (i + 1) + 0.0001, 0.5]
    subreddits = flags.groupby(["subreddit_cat", "subreddit", "subm_covid", "week"])[
        ["flag", *TYPES]
    ].sum().reset_index().assign(n_subm_all=20, n_subm_kept=10)  # fmt: skip
    for name, table in {
        "app_flags_day": flags, "app_subm_day": posts, "app_subr_week": subreddits,
    }.items():  # fmt: skip
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    return dd.load_data(tmp_path)


def _filters(**kwargs):
    defaults = {"weeks": (28, 29), "flag_types": tuple(TYPES), "min_flags": 1}
    return dd.Filters(**{**defaults, **kwargs})


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.bbc.co.uk/news/world-1", "bbc.co.uk"),
        ("http://example.com/a/b", "example.com"),
        ("not a link", "-"),
    ],
)
def test_domain_of(url, domain):
    assert dd.domain_of(url) == domain


def test_load_adds_domain_and_week_labels(tiny):
    assert tiny.submissions.set_index("subm_id").loc["a", "domain"] == "site-a.com"
    assert tiny.weeks.loc[29, "label"] == "07/14"
    assert tiny.week_range == (28, 29)
    assert tiny.year == "2020"


def test_filters(tiny):
    flags, posts = dd.filter_flags(tiny, _filters())
    assert set(posts["subm_id"]) == {"a", "b", "c"}

    _, posts = dd.filter_flags(tiny, _filters(min_flags=2))
    assert posts["subm_id"].tolist() == ["a"]

    _, posts = dd.filter_flags(tiny, _filters(covid_only=True))
    assert posts["subm_id"].tolist() == ["a"]

    _, posts = dd.filter_flags(tiny, _filters(flag_types=("bs",)))
    assert posts["subm_id"].tolist() == ["b"]

    _, posts = dd.filter_flags(tiny, _filters(weeks=(28, 28)))
    assert posts["subm_id"].tolist() == ["a"]


def test_json_round_trip_keeps_rows(tiny):
    flags, _ = dd.filter_flags(tiny, _filters())
    assert len(dd.from_json(dd.to_json(flags))) == len(flags)


def test_subreddit_filter_matches_whole_names(tiny):
    _, posts = dd.filter_flags(tiny, _filters(subreddits=("news",)))
    assert set(posts["subreddit"]) == {"news"}


def test_flagged_share_uses_posts_of_the_selection(tiny):
    filters = _filters(subreddits=("worldnews",))
    _, posts = dd.filter_flags(tiny, filters)
    share = dd.flagged_share(posts, tiny, filters).set_index("week")
    assert share.loc[29, "n_flags"] == 2  # posts b and c
    assert share.loc[29, "n_subm_kept"] == 10  # worldnews only, not all subreddits
    assert share.loc[29, "perc_FLAG_pr"] == 20.0


def test_flagged_domains_counts_posts_and_sentences(tiny):
    flags, _ = dd.filter_flags(tiny, _filters())
    table = dd.flagged_domains(flags).set_index("website")
    assert table.loc["site-a.com"].tolist() == [1, 2]
    assert table.loc["site-b.com"].tolist() == [1, 1]


def test_no_user_names_are_shown(tmp_path, tiny):
    assert dd.mask_users(pd.Series(["as u/some_user said", "see /u/Other-1!"])).tolist() == [
        "as u/[user] said",
        "see u/[user]!",
    ]
    columns = {c for table in (tiny.flags, tiny.submissions, tiny.subreddits) for c in table}
    assert not {c for c in columns if "author" in c}


def test_removed_posts_are_hidden_by_default(tmp_path, tiny):
    flags = pd.read_csv(tmp_path / "app_flags_day.csv")
    flags.loc[flags["subm_id"] == "b", "subm_removed"] = "moderator"
    flags.to_csv(tmp_path / "app_flags_day.csv", index=False)
    assert "b" not in set(dd.load_data(tmp_path).submissions["subm_id"])
    assert "b" in set(dd.load_data(tmp_path, show_removed=True).submissions["subm_id"])


def test_similar_posts_match_the_full_distance_matrix(tiny):
    """Per-post distances give the same result as the original all-pairs matrix."""
    posts = tiny.submissions
    full = pd.DataFrame(euclidean_distances(posts[dd.COMPONENTS]), index=posts["subm_id"])
    for subm_id in posts["subm_id"]:
        nearest = full.loc[subm_id].sort_values().head(20)
        expected = posts.loc[nearest[nearest < 0.005].index, "subm_id"]
        found = dd.similar_posts(tiny, subm_id)
        assert set(found.index) == set(expected.index)


def test_app_serves_layout_and_every_tab(tiny, tmp_path):
    from stroo.dashboard import layout
    from stroo.dashboard.app import create_app

    app = create_app(tmp_path)
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    for tab in layout.TABS:
        if tab not in ("method", "highlights"):
            assert getattr(layout, f"{tab}_tab")() is not None
    assert "authors" not in layout.TABS


@pytest.mark.data
def test_paper_descriptives_from_figshare_dashboard_data(data_dir):
    """Section 5 of the paper, recomputed from the published dashboard tables."""
    posts = pd.read_csv(data_dir / "dashboard" / "app_subm_day.csv").query("week <= 23")
    assert len(posts) == 33609
    assert round(posts["subm_covid"].mean(), 2) == 0.47
    assert round((posts["flag"] == 1).mean(), 2) == 0.83
    types = posts[["disinformation", "fakenews", "bs", "misleading", "unreliable", "propaganda"]]
    share = (types.sum() / types.sum().sum()).round(2)
    assert share.to_dict() == {
        "disinformation": 0.03, "fakenews": 0.23, "bs": 0.37, "misleading": 0.25,
        "unreliable": 0.0, "propaganda": 0.12,
    }  # fmt: skip
    by_category = posts["subreddit_cat"].value_counts(normalize=True)
    assert np.isclose(by_category["coronavirus"], 0.277, atol=0.001)
    assert posts.groupby("month").size().idxmax() == 3
