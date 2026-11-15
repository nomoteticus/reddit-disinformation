"""Regular expressions used before and after pattern matching (Listing 4).

All of them are applied to lower-cased text.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import settings
from .vocabulary import voc

_FLAG_NOUNS = (
    voc.obj.disinfo + voc.obj.clickbait + voc.obj.falsehood + voc.obj.propaganda + voc.obj.bs
)
_FLAG_ATTRIBUTES = (
    voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda + voc.attr.bs
)


def _word_start_regex(words) -> re.Pattern:
    """Match any of ``words`` at the start of a word (no boundary at the end)."""
    return re.compile("|".join(r"\b" + word for word in words))


#: Keyword pre-filter: a comment is parsed only if it contains one of these.
KEYWORD_REGEX = _word_start_regex(_FLAG_NOUNS + voc.obj.news + _FLAG_ATTRIBUTES)

#: Posts that are themselves about false information; their flags are dropped.
ABOUT_FAKE_NEWS_REGEX = _word_start_regex(_FLAG_NOUNS + _FLAG_ATTRIBUTES + ["conspir"])

# Sarcasm and irony: "\s" or "/s" markers, scare quotes ('"fake" news'), "jk".
_SARCASM_MARKER = r"\\s\b|\/s\b"
_SCARE_QUOTES = "|".join(
    "[*'\"“”‘’`´˝˶]" + word for word in voc.attr.false + voc.obj.disinfo + voc.attr.propaganda
)
_JOKING = r"\bjk\b"
SARCASM_REGEX = f"{_SARCASM_MARKER}|{_SCARE_QUOTES}|{_JOKING}"

# Bots.
BOT_BODY_REGEX = "([iI] am a bot)|([iI]'m a bot)|(bot just does our dirty work)"
BOT_AUTHOR_REGEX = "bot$|automod"


def load_covid_regex(keywords_file: Path | None = None) -> re.Pattern:
    """Build the COVID-19 regex from ``config/covid_keywords.txt``.

    Each line is a search term such as ``corona*``; the wildcard characters are
    dropped and a word boundary is put in front, giving ``\\bcorona``.
    """
    keywords_file = keywords_file or settings.COVID_KEYWORDS_FILE
    with open(keywords_file, encoding="utf-8") as file:
        terms = [r"\b" + re.sub(r"[*()]", "", line.rstrip()) for line in file]
    return re.compile("|".join(["china_flu"] + terms))


COVID_REGEX = load_covid_regex()
