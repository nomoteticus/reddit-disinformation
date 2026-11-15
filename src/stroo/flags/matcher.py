"""The POS matcher: applies the patterns to one sentence at a time.

Usage::

    matcher = FlagMatcher()
    for sentence in matcher.nlp("This is fake news. Great article!").sents:
        matcher.is_flag(sentence)       # True, False
        matcher.structures(sentence)    # ['svao'], []
        matcher.labels(sentence)        # ['fakenews', 'flag'], []
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

from .patterns import PATTERNS
from .vocabulary import FLAG_TYPE_KEYWORDS

SPACY_MODEL = "en_core_web_sm"
_SPACY_2 = spacy.__version__.startswith("2.")

#: Column order of the label indicators in the MATCH files.
LABELS = ["flag"] + list(FLAG_TYPE_KEYWORDS)


def _add(matcher: Matcher, name: str, patterns: list) -> None:
    """``Matcher.add`` changed its signature between spaCy 2 and 3."""
    if _SPACY_2:
        matcher.add(name, None, *patterns)
    else:
        matcher.add(name, patterns)


class FlagMatcher:
    """Three spaCy matchers sharing one vocabulary.

    ``structure``
        one rule per pattern, to see *which* pattern a sentence matches;
    ``any_pattern``
        a single rule ``flag`` that fires if any pattern matches;
    ``label``
        the ``flag`` rule plus one keyword rule per flag type.

    A flag type is assigned when its keyword appears anywhere in the sentence,
    whether or not that keyword is the one inside the matched pattern.
    """

    def __init__(self, nlp=None):
        self.nlp = nlp if nlp is not None else spacy.load(SPACY_MODEL)
        vocab = self.nlp.vocab

        self.structure = Matcher(vocab)
        for name, pattern in PATTERNS.items():
            _add(self.structure, name, [pattern])

        self.any_pattern = Matcher(vocab)
        _add(self.any_pattern, "flag", list(PATTERNS.values()))

        self.label = Matcher(vocab)
        _add(self.label, "flag", list(PATTERNS.values()))
        for flag_type, keywords in FLAG_TYPE_KEYWORDS.items():
            _add(self.label, flag_type, [[{"LOWER": {"IN": keywords}}]])

    def _names(self, matcher: Matcher, sentence: Span) -> list[str]:
        # Each sentence is matched as its own document, so that ROOT and the
        # other dependency labels refer to this sentence only.
        strings = self.nlp.vocab.strings
        return [strings[match_id] for match_id, _, _ in matcher(sentence.as_doc())]

    def structures(self, sentence: Span) -> list[str]:
        """Names of the patterns the sentence matches (one entry per match)."""
        return self._names(self.structure, sentence)

    def labels(self, sentence: Span) -> list[str]:
        """``'flag'`` if a pattern matches, plus every flag type whose keyword occurs."""
        return self._names(self.label, sentence)

    def is_flag(self, sentence: Span) -> bool:
        """Whether the sentence matches at least one pattern."""
        return len(self.any_pattern(sentence.as_doc())) > 0

    def comment_is_flag(self, text: str) -> bool:
        """Whether any sentence of a comment is a flag. Text is lower-cased first."""
        return any(self.is_flag(sentence) for sentence in self.nlp(text.lower()).sents)


@lru_cache(maxsize=1)
def get_matcher() -> FlagMatcher:
    """A shared matcher, loading the spaCy model on first use."""
    return FlagMatcher()
