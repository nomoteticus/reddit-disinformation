"""The 21 syntactic patterns of the POS matcher (Section 4.3, Table 1).

Each pattern is a list of spaCy ``Matcher`` token specifications. A pattern
name spells out the slots it expects:

    s = subject, v = verb, o = object, a = attribute, n = negation,
    i = "I", y = "you", st = "stop"

so ``svao`` matches "This is fake news" and ``naao`` matches "Not a reliable
source". Up to five (sometimes fewer) extra words are allowed between slots.
"""

from __future__ import annotations

from .vocabulary import (
    ATTRIBUTE_FLAG,
    ATTRIBUTE_FLAG_NO_BS,
    ATTRIBUTE_POSITIVE,
    OBJECT_FLAG,
    OBJECT_FLAG_NO_BS,
    OBJECT_NEUTRAL,
    SUBJECT_ANY,
    SUBJECT_POST,
    VERB_BE_FEEL,
    deps,
    pos,
    voc,
)


def _token(*, dep=None, pos_in=None, lower=None, lemma=None, op=None) -> dict:
    """Build one token specification.

    ``dep`` is a single label or a list of accepted labels; ``lower`` and
    ``lemma`` are lists of accepted word forms.
    """
    spec: dict = {}
    if dep is not None:
        spec["DEP"] = dep if isinstance(dep, str) else {"IN": dep}
    if pos_in is not None:
        spec["POS"] = {"IN": pos_in}
    if lower is not None:
        spec["LOWER"] = {"IN": lower}
    if lemma is not None:
        spec["LEMMA"] = {"IN": lemma}
    if op is not None:
        spec["OP"] = op
    return spec


def _gap(max_words: int) -> list[dict]:
    """Up to ``max_words`` optional words, none of them a negation."""
    return [{"LEMMA": {"NOT_IN": voc.neg.neg}, "OP": "?"}] * max_words


def _any_gap(max_words: int) -> list[dict]:
    """Up to ``max_words`` optional words of any kind."""
    return [{"OP": "?"}] * max_words


# --- Slots shared by several patterns ---------------------------------------

_SUBJECT_ANY = _token(dep=deps.subjects, lower=SUBJECT_ANY, op="+")
_SUBJECT_THIS = _token(dep=deps.subjects, lower=voc.subj.this, op="+")
_SUBJECT_I = _token(dep=deps.subjects, lower=voc.subj.fpers)
_SUBJECT_YOU = _token(dep=deps.subjects, lower=voc.subj.secpers)

_VERB_BE_FEEL = _token(dep=deps.predicates, lemma=VERB_BE_FEEL)
_VERB_BE = _token(dep=deps.predicates, lemma=voc.pred.be)
_VERB_FEEL_ROOT = _token(dep="ROOT", lemma=voc.pred.feel)
_VERB_CALL_ROOT = _token(dep="ROOT", lemma=voc.pred.call)
_VERB_SPREAD_ROOT = _token(dep="ROOT", lemma=voc.pred.spread)
_CLAUSE_BE_FEEL = _token(dep="ccomp", lemma=VERB_BE_FEEL)
_CLAUSE_SPREAD_FEEL = _token(dep="ccomp", lemma=voc.pred.spread + voc.pred.feel)
_VERB_STOP = _token(
    dep=["ROOT", "aux"], pos_in=["VERB", "AUX", "PROPN"], lower=voc.pred.stop + voc.pred.report
)
_VERB_SPREAD_OR_BELIEVE = _token(lemma=voc.pred.spread + voc.pred.beleive)

_OBJECT_FLAG = _token(dep=deps.objects, lemma=OBJECT_FLAG, op="+")
_OBJECT_NEUTRAL = _token(dep=deps.objects, lemma=OBJECT_NEUTRAL, op="+")
_ATTRIBUTE_FLAG = _token(dep=deps.attributes, lemma=ATTRIBUTE_FLAG, op="+")
_ATTRIBUTE_POSITIVE = _token(dep=deps.objects, lemma=ATTRIBUTE_POSITIVE, op="+")
_NEGATION = _token(dep=deps.neg, lemma=voc.neg.neg)


# --- The patterns -----------------------------------------------------------

PATTERNS: dict[str, list[dict]] = {
    # This is disinformation. Article seems like propaganda.
    "svo": [_SUBJECT_ANY, *_gap(5), _VERB_BE_FEEL, *_gap(5), _OBJECT_FLAG],
    # This is a misleading title. This has to be fake news.
    "svao": [
        _SUBJECT_ANY,
        *_gap(5),
        _VERB_BE_FEEL,
        *_gap(5),
        _ATTRIBUTE_FLAG,
        *_gap(5),
        _OBJECT_NEUTRAL,
    ],
    # Disinformation! Bullshit!
    "o": [
        _token(
            dep="ROOT",
            pos_in=pos.nouns + pos.adjs,
            lemma=OBJECT_FLAG + voc.attr.false + voc.attr.misleading,
        )
    ],
    # Fake news!
    "ao": [
        _ATTRIBUTE_FLAG,
        *_gap(5),
        _token(dep="ROOT", pos_in=pos.nouns, lemma=OBJECT_NEUTRAL, op="+"),
    ],
    # Looks like disinformation.
    "vo": [_VERB_FEEL_ROOT, *_gap(5), _OBJECT_FLAG],
    # Sounds like fake news.
    "vao": [
        _VERB_FEEL_ROOT,
        *_gap(5),
        _token(dep=deps.attributes, lemma=ATTRIBUTE_FLAG_NO_BS, op="+"),
        *_gap(5),
        _OBJECT_NEUTRAL,
    ],
    # I think this is propaganda.
    "ivsvo": [
        _SUBJECT_I,
        *_gap(3),
        _token(dep=deps.predicates, lemma=voc.pred.think),
        *_gap(3),
        _token(dep="nsubj", lemma=SUBJECT_ANY, op="+"),
        *_gap(5),
        _CLAUSE_BE_FEEL,
        *_gap(5),
        _token(dep=deps.objects, lemma=OBJECT_FLAG_NO_BS, op="+"),
    ],
    # I believe this is fake news.
    "ivsvao": [
        _SUBJECT_I,
        *_gap(3),
        _token(dep=deps.predicates, lemma=voc.pred.think),
        *_gap(3),
        _token(dep="nsubj", lower=SUBJECT_ANY, op="+"),
        *_gap(5),
        _CLAUSE_BE_FEEL,
        *_gap(5),
        _ATTRIBUTE_FLAG,
        *_gap(5),
        _OBJECT_NEUTRAL,
    ],
    # I think the title is misleading.
    "ivsva": [
        _SUBJECT_I,
        *_gap(3),
        _token(dep="ROOT", lemma=voc.pred.think),
        *_gap(3),
        _token(dep="nsubj", lower=SUBJECT_POST, op="+"),
        *_gap(5),
        _CLAUSE_BE_FEEL,
        *_gap(5),
        _ATTRIBUTE_FLAG,
    ],
    # I call bullshit.
    "ivo": [_SUBJECT_I, *_gap(3), _VERB_CALL_ROOT, *_gap(3), _OBJECT_FLAG],
    # I call fake news.
    "ivao": [
        _SUBJECT_I,
        *_gap(3),
        _VERB_CALL_ROOT,
        *_gap(3),
        _token(dep=deps.attributes, lemma=ATTRIBUTE_FLAG_NO_BS, op="+"),
        *_gap(3),
        _OBJECT_NEUTRAL,
    ],
    # Source is unreliable. Title seems false.
    "sva": [
        _SUBJECT_ANY,
        *_gap(5),
        _VERB_BE_FEEL,
        *_gap(5),
        _token(
            dep=deps.objects,
            op="+",
            lemma=voc.attr.false + voc.attr.unreliable + voc.attr.misleading,
        ),
    ],
    # Source is not reliable.
    "svna": [
        _token(dep=deps.subjects, lower=SUBJECT_POST, op="+"),
        *_any_gap(5),
        _token(dep="ROOT", lemma=VERB_BE_FEEL),
        *_any_gap(2),
        _NEGATION,
        *_any_gap(2),
        _ATTRIBUTE_POSITIVE,
    ],
    # Article does not feel real. Source may not be reliable.
    "svna2": [
        _token(dep=deps.subjects, lower=SUBJECT_POST),
        *_any_gap(5),
        _NEGATION,
        *_any_gap(3),
        _token(dep="ROOT", lemma=VERB_BE_FEEL),
        *_any_gap(3),
        _ATTRIBUTE_POSITIVE,
    ],
    # Not a reliable source.
    "naao": [
        _NEGATION,
        *_gap(5),
        _token(dep=deps.attributes, lemma=ATTRIBUTE_POSITIVE),
        *_any_gap(5),
        _token(dep=deps.objects + ["ROOT"], lower=SUBJECT_POST),
    ],
    # You are propagating disinformation.
    "yvo": [_SUBJECT_YOU, *_gap(3), _VERB_SPREAD_ROOT, *_gap(5), _OBJECT_FLAG],
    # You are spreading false information.
    "yvao": [
        _SUBJECT_YOU,
        *_gap(3),
        _VERB_SPREAD_ROOT,
        *_gap(5),
        _token(dep=deps.attributes, lemma=ATTRIBUTE_FLAG_NO_BS),
        *_gap(5),
        _OBJECT_NEUTRAL,
    ],
    # Stop spreading disinformation! Don't fall for propaganda. Reported for spreading bs.
    "stvo": [_VERB_STOP, *_any_gap(5), _VERB_SPREAD_OR_BELIEVE, *_gap(5), _OBJECT_FLAG],
    # Stop spreading false news!
    "stvao": [
        _VERB_STOP,
        *_any_gap(5),
        _VERB_SPREAD_OR_BELIEVE,
        *_gap(5),
        _token(dep=deps.attributes, lemma=ATTRIBUTE_FLAG_NO_BS),
        *_gap(5),
        _OBJECT_NEUTRAL,
    ],
    # This is how disinformation spreads.
    "svsv": [
        _SUBJECT_THIS,
        *_gap(3),
        _VERB_BE,
        *_gap(5),
        _token(dep=deps.subjects, lemma=OBJECT_FLAG, op="+"),
        *_gap(3),
        _CLAUSE_SPREAD_FEEL,
    ],
    # This is what fake news feels like.
    "svsav": [
        _SUBJECT_THIS,
        *_gap(3),
        _VERB_BE,
        *_gap(5),
        _ATTRIBUTE_FLAG,
        *_gap(5),
        _token(dep=deps.subjects, lemma=OBJECT_NEUTRAL, op="+"),
        *_gap(3),
        _CLAUSE_SPREAD_FEEL,
    ],
}

#: Written but never added to the matcher; kept for reference.
UNUSED_PATTERNS: dict[str, list[dict]] = {
    # Propaganda reported.
    "ost": [
        _token(dep=deps.subjects, lower=OBJECT_FLAG),
        *_gap(2),
        _token(dep="ROOT", lemma=voc.pred.report),
    ],
    # Fake news reported.
    "aost": [
        _token(dep=deps.attributes, lower=ATTRIBUTE_FLAG_NO_BS),
        *_gap(5),
        _token(dep=deps.subjects, lemma=OBJECT_NEUTRAL),
        *_gap(2),
        _token(dep="ROOT", lemma=voc.pred.report),
    ],
}
