"""The validation code must reproduce Tables 3 and 4 of the paper."""

import pandas as pd
import pytest

from stroo.validation import metrics, sampling

# Table 3 of the paper: prec_KEYWORD, precision_POS, recall_POS, F1_POS, n_comments.
TABLE_3 = {
    "misleading": (0.83, 0.99, 0.44, 0.61, 233),
    "propaganda": (0.51, 0.87, 0.48, 0.62, 248),
    "fakenews": (0.27, 0.73, 0.54, 0.62, 257),
    "unreliable": (0.15, 0.67, 0.36, 0.47, 204),
    "disinformation": (0.29, 0.65, 0.32, 0.43, 216),
    "bs": (0.23, 0.52, 0.50, 0.51, 241),
}
# Table 4 of the paper, column "POS Matcher".
TABLE_4_POS = {"accuracy": 0.72, "precision": 0.73, "recall": 0.45, "F1": 0.55, "AUC-ROC": 0.67}


@pytest.mark.data
def test_table_2_valid_counts(data_dir):
    train = metrics.valid(metrics.load_labelled(metrics.TRAIN_FILE, data_dir / "labelled"))
    test = metrics.valid(metrics.load_labelled(metrics.TEST_FILE, data_dir / "labelled"))
    assert train["sample"].value_counts().to_dict() == {"pos": 591, "all": 573}
    assert len(test) == 291
    assert len(train) + len(test) == 1455


@pytest.mark.data
def test_table_3_is_reproduced(data_dir):
    table = metrics.keyword_vs_pos(metrics.load_labelled(metrics.TRAIN_FILE, data_dir / "labelled"))
    for flag_type, expected in TABLE_3.items():
        row = table.loc[flag_type]
        assert tuple(row.iloc[:4].round(2)) == expected[:4], flag_type
        assert row["n_comments"] == expected[4], flag_type


@pytest.mark.data
def test_table_4_pos_matcher_column_is_reproduced(data_dir):
    test = metrics.load_labelled(metrics.TEST_FILE, data_dir / "labelled")
    result = metrics.pos_matcher_on_test_set(test)
    assert {name: round(value, 2) for name, value in result.items()} == TABLE_4_POS


def test_tokenizer_drops_stop_words_and_punctuation():
    from stroo.validation.ml_baselines import tokenize

    assert tokenize("This is FAKE news, 100%!") == ["fake", "news"]


def test_draw_samples_sizes_and_strata():
    rows = []
    for i in range(400):
        rows.append(
            {"comm_body": f"comment {i}", "subm_title": "t", "domain": "d.com",
             "flag": i % 2, "disinformation": 1, "fakenews": 1 + i % 3, "bs": 1,
             "misleading": 1, "unreliable": 1, "propaganda": 1}
        )  # fmt: skip
    candidates = pd.DataFrame(rows)
    samples = sampling.draw_samples(candidates, n_keyword=10, n_pos=5)
    assert len(samples) == 6 * 15
    counts = samples.groupby(["word", "sample"]).size()
    assert (counts.xs("all", level="sample") == 10).all()
    assert (counts.xs("pos", level="sample") == 5).all()
    assert (samples.loc[samples["sample"] == "pos", "flag"] > 0).all()
    # Same seed, same sample.
    again = sampling.draw_samples(candidates, n_keyword=10, n_pos=5)
    assert samples["comm_body"].tolist() == again["comm_body"].tolist()


@pytest.mark.data
def test_section_5_5_post_level_figures_are_reproduced(data_dir):
    from stroo.validation import factcheck

    path = data_dir / "factcheck" / "SSD.csv"
    if not path.exists():
        pytest.skip("SSD.csv not available")
    result = factcheck.compare_with_suspicious(pd.read_csv(path))
    assert result["posts from suspicious domains"] == 23076
    assert result["posts from suspicious domains flagged"] == 830
    assert round(100 * result["share of posts from suspicious domains flagged"], 1) == 3.6
    assert round(100 * result["share of posts from other domains flagged"], 2) == 2.83
    assert round(100 * result["share of other domains flagged"]) == 7
