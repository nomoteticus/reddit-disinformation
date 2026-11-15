"""Machine-learning baselines the POS matcher is compared with (Section 4.4.3).

Two classifiers are trained on the 573 valid comments sampled by keyword from
January-July 2020 and evaluated on the 291 valid comments from August-October
2020, the same test set as the POS matcher:

* random forest;
* logistic regression with an L1 or L2 penalty.

Features are TF-IDF weights of lower-cased, lemmatised words without stop
words. Hyper-parameters are chosen by 5-fold cross-validation on F1.

This follows the notebook published with the paper
(doi:10.6084/m9.figshare.13174145). The exact figures depend on the versions of
spaCy (lemmatiser) and scikit-learn in use.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS

from .metrics import TEST_FILE, TRAIN_FILE, classification_report, load_labelled, valid

SEED = 777
MAX_FEATURES = (50, 250, 500)

RANDOM_FOREST_GRID = {
    "vectorizer__max_features": MAX_FEATURES,
    "classifier__n_estimators": (50, 100, 300),
    "classifier__max_depth": (10, 30, 50),
}
LOGISTIC_GRID = {
    "vectorizer__max_features": MAX_FEATURES,
    "classifier__C": np.logspace(-3, 3, 11),
    "classifier__penalty": ("l1", "l2"),
}


@lru_cache(maxsize=1)
def _lemmatiser():
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def tokenize(text: str) -> list[str]:
    """Lower-cased lemmas of a comment, without stop words, numbers or punctuation."""
    lemmas = (token.lemma_.lower().strip() for token in _lemmatiser()(text))
    return [lemma for lemma in lemmas if lemma not in STOP_WORDS and lemma.isalpha()]


def _strip_and_lower(text: str) -> str:
    return text.strip().lower()


def make_pipeline(classifier) -> Pipeline:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, preprocessor=_strip_and_lower, token_pattern=None
    )
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def load_train_test(labelled_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valid comments sampled by keyword: January-July (train), August-October (test)."""
    train = valid(load_labelled(TRAIN_FILE, labelled_dir))
    test = valid(load_labelled(TEST_FILE, labelled_dir))
    return train[train["sample"] == "all"], test[test["sample"] == "all"]


def fit_and_evaluate(
    classifier, grid: dict, train: pd.DataFrame, test: pd.DataFrame, n_jobs: int = 1
) -> dict[str, float]:
    """Tune on the training set by cross-validation, then score on the test set."""
    search = GridSearchCV(make_pipeline(classifier), grid, scoring="f1", n_jobs=n_jobs)
    search.fit(train["comm_body"], train["is_flag"])
    predicted = search.best_estimator_.predict(test["comm_body"])
    return classification_report(test["is_flag"], predicted)


def compare(labelled_dir: Path | None = None, n_jobs: int = 1) -> pd.DataFrame:
    """Table 4: POS matcher, logistic regression and random forest on the test set."""
    train, test = load_train_test(labelled_dir)
    results = {
        "POS matcher": classification_report(test["is_flag"], test["matches_POS"]),
        "L1/L2 logistic regression": fit_and_evaluate(
            LogisticRegression(random_state=SEED), LOGISTIC_GRID, train, test, n_jobs
        ),
        "Random forests": fit_and_evaluate(
            RandomForestClassifier(random_state=SEED), RANDOM_FOREST_GRID, train, test, n_jobs
        ),
    }
    table = pd.DataFrame(results)
    table.loc["n_comments"] = len(test)
    return table
