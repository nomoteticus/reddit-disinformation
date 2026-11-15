"""Offline tests of the helpers that select and tidy scraped data."""

import pandas as pd

from stroo.collect import cleaning


def _submissions():
    return pd.DataFrame(
        {
            "id": ["t3_a", "t3_b", "t3_c", "t3_d", "t3_e"],
            "is_self": [False, True, False, False, False],
            "post_hint": ["link", "self", "image", "link", None],
            "domain": ["bbc.co.uk", "self.news", "i.redd.it", "youtube.com", "reuters.com"],
            "num_comments": [5, 3, 9, 2, 0],
        }
    )


def test_only_external_news_links_with_comments_are_scraped():
    assert cleaning.news_link_ids(_submissions()).tolist() == ["t3_a"]


def test_already_scraped_submissions_are_excluded():
    assert cleaning.news_link_ids(_submissions(), exclude={"t3_a"}).empty


def test_first_and_second_level_comments_only():
    comments = [
        {"id": "1", "parent_id": "t3_a"},
        {"id": "2", "parent_id": "t1_1"},  # reply to a first-level comment
        {"id": "3", "parent_id": "t1_2"},  # third level: dropped
    ]
    assert [c["id"] for c in cleaning.first_and_second_level(comments)] == ["1", "2"]


def test_clean_comments_caps_per_submission_and_tidies_text():
    n = cleaning.MAX_COMMENTS_PER_SUBMISSION + 5
    comments = pd.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "created": range(1_600_000_000, 1_600_000_000 + n),
            "subreddit": "news",
            "link_id": "t3_a",
            "parent_id": "t3_a",
            "body": "two\nlines   here",
            "edited": False,
            "author": "someone",
            "collapsed": False,
            "no_follow": True,
            "score": 1,
            "distinguished": None,
        }
    )
    cleaned = cleaning.clean_comments(comments)
    assert len(cleaned) == cleaning.MAX_COMMENTS_PER_SUBMISSION
    assert cleaned["id"].iloc[0] == "t1_0"  # earliest comments are kept
    assert set(cleaned["body"]) == {"two lines here"}


def test_merge_new_rows_prefers_new_values():
    old = pd.DataFrame({"id": ["a", "b"], "score": [1, 1], "created": [1, 2]})
    new = pd.DataFrame({"id": ["b", "c"], "score": [9, 9], "created": [2, 3]})
    merged = cleaning.merge_new_rows(new, old, sort_by="created")
    assert merged["id"].tolist() == ["c", "b", "a"]
    assert merged.set_index("id")["score"].to_dict() == {"a": 1, "b": 9, "c": 9}


def test_chunked():
    assert [list(c) for c in cleaning.chunked([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]


def test_posts_per_day_keeps_busy_subreddits():
    from stroo.collect.subreddits import posts_per_day

    days = pd.date_range("2020-04-01", periods=10).strftime("%Y-%m-%d")
    rows = [
        {"id": f"{sub}{day}{i}", "subreddit": sub, "created_utc": day}
        for day in days
        for sub, n in (("busy", 12), ("quiet", 3))
        for i in range(n)
    ]
    table = posts_per_day(pd.DataFrame(rows))
    assert table.index.tolist() == ["busy"]
    assert table.loc["busy", "total"] == 12
    assert table.loc["busy", "total7d"] == 12
