"""Check that `stroo.flags` reproduces the 2020 matching results exactly.

Run this in an environment with the library versions of the paper
(Python 3.8, spaCy 2.3.5, en_core_web_sm 2.3.1; see environment-paper.yml):

    python scripts/check_paper_reproduction.py

It needs two local files that are not part of the repository:

* data/baseline/july_sample_comments.csv   - comments (columns: id, body)
* data/baseline/orig2020_MATCH_july_sample.csv - the 2020 output for them
* data/labelled/classified_regex_labeled_test.csv - annotated test set (Figshare)
"""

import sys
from pathlib import Path

import pandas as pd
import spacy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stroo.flags.detect import match_comments, prefilter_comments  # noqa: E402
from stroo.flags.matcher import get_matcher  # noqa: E402

DATA = REPO / "data"
matcher = get_matcher()
print("spaCy", spacy.__version__, "| model", matcher.nlp.meta["version"])

comments = prefilter_comments(pd.read_csv(DATA / "baseline" / "july_sample_comments.csv"))
new = match_comments(comments["id"], comments["body"], matcher)
old = pd.read_csv(DATA / "baseline" / "orig2020_MATCH_july_sample.csv")
new_flagged, old_flagged = set(new.comm_id[new.flag > 0]), set(old.comm_id[old.flag > 0])
print(
    f"July 2020 sample - flagged comments: 2020 run {len(old_flagged)}, "
    f"now {len(new_flagged)}, identical {len(new_flagged & old_flagged)}"
)

test = pd.read_csv(DATA / "labelled" / "classified_regex_labeled_test.csv")
found = match_comments(test["id"], test["comm_body"].astype(str).str.lower(), matcher)
now = test["id"].isin(set(found.comm_id[found.flag > 0])).astype(int)
print(
    f"Annotated test set - agreement with published matches_POS: "
    f"{(now == test['matches_POS']).mean():.1%} of {len(test)}"
)
