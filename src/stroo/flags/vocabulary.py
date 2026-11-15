"""Vocabulary of the informal flag (Appendix B, Listings 1 and 2 of the paper).

The class and attribute names (``voc.subj.article``, ``deps.objects``, ...) are
kept exactly as printed in the paper so the appendix can be read against the
code. The word lists are reproduced verbatim from the appendix.
"""

from __future__ import annotations


class pos:  # noqa: N801 - name follows the paper
    """Part-of-speech tags."""

    nouns = ["NOUN", "PROPN"]
    adjs = ["ADJ"]


class deps:  # noqa: N801 - name follows the paper
    """Dependency labels."""

    root = ["ROOT"]
    predicates = ["ROOT", "xcomp"]
    subjects = ["nsubj", "nsubjpass"]
    objects = ["dobj", "pobj", "attr", "acomp", "oprd", "conj", "compound", "nsubjpass"]
    attributes = ["amod"] + objects
    neg = ["neg", "det"]


class voc:  # noqa: N801 - name follows the paper
    """All keywords, grouped by the grammatical slot they fill."""

    class subj:  # noqa: N801
        # Nouns
        article = ["article", "submission", "sub", "post"]
        title = ["title", "headline", "header"]
        source = ["source", "website", "site", "url", "link"]
        # Pronouns
        fpers = ["i", "me", "we"]
        secpers = ["you"]
        this = ["it", "this", "that", "here"]

    class pred:  # noqa: N801
        be = ["be"]
        feel = ["look", "sound", "feel", "seem", "smell", "stink"]
        think = ["guess", "think", "beleive", "say", "know", "feel", "suspect"]
        beleive = ["beleive", "fall", "buy"]
        call = ["call"]
        stop = ["stop", "quit", "refrain", "do"]
        report = ["flag", "report", "remove"]
        spread = ["spread", "propagate", "spew", "distribute", "promote", "post", "submit"]

    class obj:  # noqa: N801
        news = ["news", "information", "info"]
        disinfo = ["disinformation", "misinformation", "malinformation"]
        clickbait = ["clickbait", "innacuracy", "innacuracies"]
        falsehood = ["falsehood", "fabrication"]
        bs = ["bullshit", "bs"]
        propaganda = ["propaganda"]

    class attr:  # noqa: N801
        false = [
            "fake",
            "false",
            "bogus",
            "fabricated",
            "manipulated",
            "manipulative",
            "inaccurate",
        ]
        misleading = [
            "misleading",
            "mislead",
            "editorialized",
            "editorialize",
            "clickbait",
            "sensationalized",
            "sensationalize",
            "sensationalist",
        ]
        unreliable = ["untrustworthy", "unreliable", "unverified"]
        bs = ["bullshit", "bs"]
        propaganda = ["propaganda"]
        real = ["real", "true", "correct"]
        reliable = ["reliable", "verified", "credible"]

    class neg:  # noqa: N801
        neg = ["not", "no"]


# --- Word groups that recur across patterns ---------------------------------

#: Things a flag can be about: "this", the post, its title, its source.
SUBJECT_ANY = voc.subj.this + voc.subj.article + voc.subj.title + voc.subj.source
#: The same without the pronouns.
SUBJECT_POST = voc.subj.article + voc.subj.title + voc.subj.source

#: Nouns that are an accusation on their own ("propaganda", "bullshit").
OBJECT_FLAG = (
    voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood
)
OBJECT_FLAG_NO_BS = voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.falsehood
#: Neutral nouns that need a negative attribute ("fake news", "misleading title").
OBJECT_NEUTRAL = voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title

ATTRIBUTE_FLAG = (
    voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda + voc.attr.bs
)
ATTRIBUTE_FLAG_NO_BS = (
    voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda
)
#: Positive attributes; a flag only when negated ("not a reliable source").
ATTRIBUTE_POSITIVE = voc.attr.real + voc.attr.reliable

VERB_BE_FEEL = voc.pred.be + voc.pred.feel

#: The six types of informal flag and the keywords that define each
#: (Listing 5 of the paper). Order is the column order of the MATCH files.
FLAG_TYPE_KEYWORDS = {
    "disinformation": voc.obj.disinfo,
    "fakenews": voc.obj.falsehood + voc.attr.false,
    "bs": voc.obj.bs,
    "misleading": voc.attr.misleading + voc.obj.clickbait,
    "unreliable": voc.attr.unreliable,
    "propaganda": voc.obj.propaganda,
}
FLAG_TYPES = list(FLAG_TYPE_KEYWORDS)
