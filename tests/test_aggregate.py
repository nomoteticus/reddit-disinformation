"""The rewritten aggregation must produce the same tables as 22_bring_together.py.

Reference outputs were produced by running the original script, unmodified,
with pandas 1.1.5 on July 2020 data split into two pseudo-months: once from
scratch on the first ("run A") and once incrementally after adding the second
("run B"). The Universal Sentence Encoder was replaced by the deterministic
stand-in in ``fake_encoder.py`` on both sides.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from fake_encoder import fake_encode
from stroo.aggregate import tables

YEAR = 2021  # the pseudo-months are named SUBM_2021_06 / _07, as the original expects
OUTPUTS = {
    "UNITED_2021_FLAG.csv": ["subreddit", "subm_id", "comm_id", "sent_id"],
    "UNITED_2021_SUBM.csv": tables.BY_SUBREDDIT_WEEK,
    "UNITED_2021_AUTH.csv": tables.BY_AUTHOR_WEEK,
    "UNITED_2021_DOM.csv": tables.BY_DOMAIN_WEEK,
    "app_flags_day.csv": ["subm_id", "sent_id", "sent"],
    "app_subm_day.csv": ["subm_id"],
    "app_subr_week.csv": tables.BY_SUBREDDIT_WEEK,
    "app_authFlaggers_week.csv": tables.BY_SUBREDDIT_DAY + ["comm_author"],
    "app_authFlagged_week.csv": tables.BY_SUBREDDIT_DAY + ["subm_author"],
    "app_domain_week.csv": tables.BY_DOMAIN_WEEK,
}


def _load(path, keys):
    table = pd.read_csv(path, lineterminator="\n", low_memory=False)
    table = table.drop(columns=[c for c in table.columns if c == "index"])
    return table.sort_values(
        keys + ["sent"] if "sent" in table and "sent" not in keys else keys, kind="stable"
    ).reset_index(drop=True)


def _assert_same(new_path, old_path, keys):
    new, old = _load(new_path, keys), _load(old_path, keys)
    assert list(new.columns) == list(old.columns)
    assert len(new) == len(old)
    components = [c for c in new.columns if c.startswith("PC")]
    for column in new.columns:
        if column in components:
            continue
        if pd.api.types.is_numeric_dtype(old[column]) or pd.api.types.is_bool_dtype(old[column]):
            np.testing.assert_allclose(
                new[column].astype(float), old[column].astype(float), err_msg=column
            )
        else:
            assert new[column].fillna("<NA>").astype(str).tolist() == (
                old[column].fillna("<NA>").astype(str).tolist()
            ), column
    # PCA: identical up to the sign of each component and numerical noise.
    for column in components:
        a, b = new[column].to_numpy(), old[column].to_numpy()
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-6, column


@pytest.mark.data
def test_identical_to_original_fresh_and_incremental(data_dir, tmp_path):
    reference = data_dir / "baseline" / "aggregate"
    raw, processed, dashboard = tmp_path / "raw", tmp_path / "processed", tmp_path / "dashboard"
    raw.mkdir()
    processed.mkdir()

    def add_month(month):
        for kind, folder in (("SUBM", raw), ("COMM", raw), ("MATCH", processed)):
            name = f"{kind}_{YEAR}_{month}.csv"
            shutil.copy(reference / "inputs" / name, folder / name)

    for month, run in (("06", "legacy_runA"), ("07", "legacy_runB")):
        add_month(month)
        tables.run(
            YEAR,
            encode=fake_encode,
            raw_dir=raw,
            processed_dir=processed,
            dashboard_dir=dashboard,
            subreddits_file=reference / "inputs" / "subr_classification.csv",
        )
        for name, keys in OUTPUTS.items():
            folder = processed if name.startswith("UNITED") else dashboard
            old_name = name if name.startswith("UNITED") else name.replace("app_", "app_2021_")
            _assert_same(folder / name, reference / run / old_name, keys)


def test_week_53_is_recoded_to_zero():
    dates = pd.Series(pd.to_datetime(["2020-12-31", "2021-01-01", "2021-01-04", "2020-07-14"]))
    assert tables.week_of_year(dates).tolist() == [0, 0, 1, 29]


def test_other_marks_flags_without_a_type_keyword():
    matches = pd.DataFrame(
        {"flag": [1, 1, 0], "disinformation": [0, 0, 0], "fakenews": [1, 0, 0], "bs": 0,
         "misleading": 0, "unreliable": 0, "propaganda": 0}
    )  # fmt: skip
    assert tables.prepare_matches(matches)["other"].tolist() == [0, 1, 0]
