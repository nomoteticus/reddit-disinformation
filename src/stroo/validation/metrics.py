"""Precision and recall of the POS matcher against manual labels (Section 4.4).

Input is the annotated-comment files published on Figshare
(doi:10.6084/m9.figshare.13315259). Their relevant columns:

``word``         keyword group the comment was sampled for (one of six flag types)
``sample``       ``all`` = sampled from keyword matches, ``pos`` = from POS matches
``matches_POS``  1 if the comment matched a pattern when it was sampled
``consensus``    final label: ``f`` flag, ``n`` not a flag, anything else uncertain
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn import metrics

from .. import settings
from ..flags import FLAG_TYPES

TRAIN_FILE = "classified_regex_labeled_train.csv"  # sets 1 and 2: January-July 2020
TEST_FILE = "classified_regex_labeled_test.csv"  # set 3: August-October 2020


def load_labelled(file_name: str, labelled_dir: Path | None = None) -> pd.DataFrame:
    """All annotated comments of one file, with ``is_flag`` (missing if uncertain)."""
    labelled = pd.read_csv((labelled_dir or settings.LABELLED_DIR) / file_name)
    is_flag = labelled["consensus"].map({"f": 1, "n": 0})
    return labelled.assign(is_flag=is_flag)


def valid(labelled: pd.DataFrame) -> pd.DataFrame:
    """Drop the comments the coders left as uncertain."""
    labelled = labelled[labelled["is_flag"].notna()]
    return labelled.assign(is_flag=labelled["is_flag"].astype(int))


def keyword_vs_pos(labelled: pd.DataFrame) -> pd.DataFrame:
    """Table 3: precision of the keyword filter; precision, recall, F1 of the POS matcher.

    ``labelled`` is the January-July file (sets 1 and 2).

    * keyword precision: share of flags among comments sampled by keyword;
    * POS precision: share of flags among all comments matching a pattern;
    * POS recall: share of the flags in the keyword sample that match a pattern;
    * ``n_comments``: annotated comments containing the keyword, uncertain ones included.
    """
    rows = {}
    for flag_type, group in valid(labelled).groupby("word"):
        keyword_sample = group[group["sample"] == "all"]
        precision = group.loc[group["matches_POS"] == 1, "is_flag"].mean()
        recall = keyword_sample.loc[keyword_sample["is_flag"] == 1, "matches_POS"].mean()
        rows[flag_type] = {
            "prec_KEYWORD": keyword_sample["is_flag"].mean(),
            "precision_POS": precision,
            "recall_POS": recall,
            "F1_POS": 2 * precision * recall / (precision + recall),
            "n_comments": int((labelled[flag_type] > 0).sum()),
        }
    return pd.DataFrame(rows).T.loc[FLAG_TYPES].astype({"n_comments": int})


def classification_report(observed, predicted) -> dict[str, float]:
    """The five diagnostics of Table 4."""
    return {
        "accuracy": metrics.accuracy_score(observed, predicted),
        "precision": metrics.precision_score(observed, predicted),
        "recall": metrics.recall_score(observed, predicted),
        "F1": metrics.f1_score(observed, predicted),
        "AUC-ROC": metrics.roc_auc_score(observed, predicted),
    }


def pos_matcher_on_test_set(test: pd.DataFrame) -> dict[str, float]:
    """Table 4, first column: the POS matcher on the August-October test set."""
    test = valid(test)
    test = test[test["sample"] == "all"]
    return classification_report(test["is_flag"], test["matches_POS"])


#: Pattern groups of Table 1 of the paper.
TABLE_1_GROUPS = {
    "o, ao": ["o", "ao"],
    "svo, svao": ["svo", "svao"],
    "sva, svna, svna2": ["sva", "svna", "svna2"],
    "naao": ["naao"],
    "vo, vao": ["vo", "vao"],
    "stvo, stvao": ["stvo", "stvao"],
    "ivo, ivao": ["ivo", "ivao"],
    "yvo, yvao": ["yvo", "yvao"],
    "ivsva, ivsvo, ivsvao": ["ivsva", "ivsvo", "ivsvao"],
    "svsv, svsav": ["svsv", "svsav"],
}


def pattern_frequencies(sentences, matcher=None) -> pd.Series:
    """Table 1: share of pattern matches by group of patterns.

    ``sentences`` are matched sentences (the ``sent`` column of the MATCH
    files). Each is parsed again and every pattern it matches is counted.
    """
    from ..flags import get_matcher

    matcher = matcher or get_matcher()
    counts: dict[str, int] = {}
    for doc in matcher.nlp.pipe(sentences, disable=["ner"]):
        for sentence in doc.sents:
            for name in matcher.structures(sentence):
                counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return pd.Series(
        {group: sum(counts.get(name, 0) for name in names) / total
         for group, names in TABLE_1_GROUPS.items()}
    )  # fmt: skip
