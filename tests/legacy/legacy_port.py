"""Load the ORIGINAL matcher module as a behaviour reference.

``functions_pos_match_original.py`` is a verbatim copy of the file as it was
on ``archive/pre-cleanup``. Only two things are patched at load time, neither
of which changes what it matches:

* ``Matcher.add(name, None, p1, p2)`` -> ``Matcher.add(name, [p1, p2])``
  (the spaCy 3 signature; skipped on spaCy 2);
* the hard-coded project root and keyword-file path.
"""
import re
import types
from pathlib import Path

import spacy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_legacy():
    src = (HERE / "functions_pos_match_original.py").read_text(encoding="utf-8")
    if not spacy.__version__.startswith("2."):
        src = re.sub(
            r'\.add\(("[a-z0-9]+"),\s*None,\s*(.*?)\)\s*\n(?=\S|\n)',
            lambda m: f".add({m.group(1)}, [{m.group(2)}])\n",
            src,
            flags=re.S,
        )
    src = src.replace(
        "rootfold = re.match('^.*reddit-disinformation', "
        "os.path.dirname(os.path.realpath(__file__))).group(0)",
        "rootfold = ''",
    )
    keywords = (REPO / "config" / "covid_keywords.txt").as_posix()
    src = src.replace("open(rootfold+'/input/keywords_covid.txt','r')", f"open('{keywords}','r')")
    module = types.ModuleType("functions_pos_match_original")
    exec(compile(src, "functions_pos_match_original", "exec"), module.__dict__)
    return module
