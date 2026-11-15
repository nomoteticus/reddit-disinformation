"""The refactored flag detection must behave exactly like the original module."""

from pathlib import Path

import pandas as pd
import pytest

from stroo.flags import PATTERNS, filters
from stroo.flags.detect import match_comments, prefilter_comments
from stroo.flags.patterns import UNUSED_PATTERNS

EXAMPLES = Path(__file__).parent / "data"

# Example phrases from Table 1 of the paper, plus the counter-examples of Section 4.3.
TABLE_1 = [
    "Fake news!",
    "Clickbait!",
    "The whole post is disinformation.",
    "This has to be fake news.",
    "Title is misleading.",
    "Source is not reliable.",
    "Not a reliable source.",
    "Not a correct headline!",
    "Looks like false news.",
    "Smells like propaganda",
    "Quit spreading misinformation.",
    "Stop posting fake news.",
    "I call bullshit.",
    "I'm calling bs.",
    "You are spreading falsehoods.",
    "You're posting false information.",
    "I think this is propaganda.",
    "I know that the title is false.",
    "This is how fake news spreads.",
    "This is what fake news looks like.",
]
NOT_FLAGS = [
    "This is great news.",
    "Post is not false.",
    "Fake news is an overused trope",
    "They doubt that the source is unreliable",
    "propaganda is bad",
]
LABELS = ["flag", "disinformation", "fakenews", "bs", "misleading", "unreliable", "propaganda"]


def _normalise(pattern):
    """Token specs with IN/NOT_IN lists turned into sets (spaCy ignores their order)."""

    def norm(value):
        if isinstance(value, dict):
            return tuple(sorted((k, frozenset(v)) for k, v in value.items()))
        return value

    return [tuple(sorted((key, norm(value)) for key, value in token.items())) for token in pattern]


def test_all_21_patterns_are_registered():
    assert len(PATTERNS) == 21


@pytest.mark.parametrize("name", [*PATTERNS, *UNUSED_PATTERNS])
def test_pattern_definitions_identical_to_original(legacy, name):
    new = {**PATTERNS, **UNUSED_PATTERNS}[name]
    assert _normalise(new) == _normalise(getattr(legacy, f"pattern_{name}"))


def test_pattern_registration_order_identical(legacy):
    original = [legacy.nlp.vocab.strings[key] for key in legacy.structurematcher._patterns]
    assert list(PATTERNS) == original


def test_regexes_identical_to_original(legacy):
    assert filters.KEYWORD_REGEX.pattern == legacy.regex_flag.pattern
    assert filters.ABOUT_FAKE_NEWS_REGEX.pattern == legacy.fake_regex.pattern
    assert filters.SARCASM_REGEX == legacy.sarcasm_and_irony_regex
    assert filters.BOT_BODY_REGEX == legacy.bot_body_regex
    assert filters.BOT_AUTHOR_REGEX == legacy.bot_author_regex
    assert filters.COVID_REGEX.pattern == legacy.covid_regex.pattern


def _sentences():
    unmatched = (EXAMPLES / "unmatched_examples.txt").read_text(encoding="utf-8").splitlines()
    return TABLE_1 + NOT_FLAGS + [line for line in unmatched if line.strip()]


@pytest.mark.parametrize("text", _sentences())
def test_matcher_identical_to_original_on_examples(legacy, matcher, text):
    new_doc, old_doc = matcher.nlp(text.lower()), legacy.nlp(text.lower())
    for new, old in zip(new_doc.sents, old_doc.sents, strict=True):
        assert matcher.structures(new) == old._.structurematched
        assert matcher.labels(new) == old._.flagtypematched
        assert matcher.is_flag(new) == old._.isflagmatched


def test_clear_cases(matcher):
    assert matcher.comment_is_flag("This is fake news.")
    assert matcher.comment_is_flag("The title is misleading.")
    assert not matcher.comment_is_flag("This is great news.")
    assert not matcher.comment_is_flag("Fake news is an overused trope.")


def test_prefilter_drops_sarcasm_and_irrelevant_comments():
    comments = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "body": ["This is FAKE news", "nice weather", "sure, 'fake' news", "propaganda /s"],
        }
    )
    assert prefilter_comments(comments)["id"].tolist() == ["a"]


def _legacy_match(legacy, ids, bodies):
    """The sentence loop of the original 21_sent_pos_df.py."""
    rows = []
    for comment_id, doc in zip(ids, legacy.nlp.pipe(bodies, disable=["ner"]), strict=True):
        for sent_id, sent in enumerate(doc.sents):
            matched = sent._.flagtypematched
            if matched:
                row = dict.fromkeys(matched, 1)
                row.update(comm_id=comment_id, sent_id=sent_id, sent=sent.text)
                rows.append(row)
    out = pd.DataFrame(rows, columns=["comm_id", "sent_id", "sent"] + LABELS)
    out[LABELS] = out[LABELS].fillna(0).astype("int64")
    return out


@pytest.mark.data
@pytest.mark.parametrize(
    "name", ["classified_regex_labeled_train.csv", "classified_regex_labeled_test.csv"]
)
def test_identical_to_original_on_annotated_comments(legacy, matcher, data_dir, name):
    labelled = pd.read_csv(data_dir / "labelled" / name)
    bodies = labelled["comm_body"].astype(str).str.lower()
    new = match_comments(labelled["id"], bodies, matcher)
    old = _legacy_match(legacy, labelled["id"], bodies)
    pd.testing.assert_frame_equal(new, old)


@pytest.mark.data
def test_identical_to_original_on_july_2020_sample(legacy, matcher, data_dir):
    comments = pd.read_csv(data_dir / "baseline" / "july_sample_comments.csv")
    comments = prefilter_comments(comments).head(3000)
    new = match_comments(comments["id"], comments["body"], matcher)
    old = _legacy_match(legacy, comments["id"], comments["body"])
    assert len(new) > 500
    pd.testing.assert_frame_equal(new, old)
