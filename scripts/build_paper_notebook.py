"""Generate and execute ``notebooks/paper_results.ipynb``.

The notebook is kept as a generated file so that its cells stay short and in
sync with the package: edit the cells below, then run

    python scripts/build_paper_notebook.py

It needs the local ``data/`` folder (see the first cell of the notebook).
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "paper_results.ipynb"

CELLS = [
    (
        "md",
        """# Results of the paper, recomputed

Achimescu & Chachev (2021), *Raising the Flag: Monitoring User Perceived Disinformation on
Reddit*, Information 12(1), 4. https://doi.org/10.3390/info12010004

Each section recomputes one table or set of figures of the paper with the functions of the
`stroo` package.

**Data needed** (none of it is in the repository; paths are relative to `data/`):

| Folder | Files | Source |
|---|---|---|
| `labelled/` | `classified_regex_labeled_train.csv`, `classified_regex_labeled_test.csv` | Figshare, doi:10.6084/m9.figshare.13315259 |
| `dashboard/` | the six `app_*.csv` tables | Figshare, doi:10.6084/m9.figshare.13174136 |
| `factcheck/` | `SSD.csv` | Figshare, doi:10.6084/m9.figshare.13174145 |
| `processed/` | `MATCH_2020_01.csv` ... `MATCH_2020_06.csv` | pipeline output (not public; Table 1 and Table 5 only) |

Sections 1 and 3 parse text and train models, so their figures depend on the versions of
spaCy and scikit-learn installed.""",
    ),
    (
        "code",
        """import warnings

import pandas as pd

from stroo import settings
from stroo.validation import factcheck, metrics

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
DATA = settings.DATA_DIR""",
    ),
    (
        "md",
        """## 1. Table 1 - frequency of the patterns

Share of pattern matches by pattern group, over the sentences flagged from January to June
2020.""",
    ),
    (
        "code",
        """matches = pd.concat(
    pd.read_csv(DATA / "processed" / f"MATCH_2020_{month:02d}.csv", lineterminator="\\n")
    for month in range(1, 7)
)
flagged_sentences = matches.loc[matches["flag"] > 0, "sent"].astype(str)
print(f"{len(matches):,} sentences with a flag keyword, {len(flagged_sentences):,} match a pattern")
metrics.pattern_frequencies(flagged_sentences).round(3).rename("share of matches").to_frame()""",
    ),
    (
        "md",
        """## 2. Tables 2 and 3 - manual annotation, keyword filter vs POS matcher

Computed from the annotated comments published on Figshare.""",
    ),
    (
        "code",
        """train = metrics.load_labelled(metrics.TRAIN_FILE)
test = metrics.load_labelled(metrics.TEST_FILE)
valid_train, valid_test = metrics.valid(train), metrics.valid(test)

table_2 = pd.DataFrame({
    "Set 1 (Jan-Jul, keyword matches)": [600, (valid_train["sample"] == "all").sum()],
    "Set 2 (Jan-Jul, POS matches)": [600, (valid_train["sample"] == "pos").sum()],
    "Set 3 (Aug-Oct, keyword matches)": [len(test), len(valid_test)],
}, index=["N", "valid N"])
table_2["Total"] = table_2.sum(axis=1)
table_2""",
    ),
    ("code", """metrics.keyword_vs_pos(train).round(2)"""),
    (
        "md",
        """## 3. Table 4 - POS matcher vs machine-learning baselines

Random forest and logistic regression are trained on the January-July comments and all three
methods are scored on the August-October test set. This cell takes a few minutes.""",
    ),
    (
        "code",
        """from stroo.validation import ml_baselines

ml_baselines.compare(n_jobs=4).round(2)""",
    ),
    (
        "md",
        """## 4. Section 5 - descriptive statistics

Flagged posts of weeks 1-23 of 2020 in the dashboard table.""",
    ),
    (
        "code",
        """posts = pd.read_csv(DATA / "dashboard" / "app_subm_day.csv").query("week <= 23")
print(f"Flagged posts: {len(posts):,}")
print(f"COVID-related: {posts['subm_covid'].mean():.0%}")

times_flagged = posts["flag"].clip(upper=3).value_counts(normalize=True).sort_index()
times_flagged.index = ["once", "twice", "three times or more"]
print("\\nTimes flagged:")
print(times_flagged.round(3).to_string())""",
    ),
    (
        "code",
        """flag_types = ["disinformation", "fakenews", "bs", "misleading", "unreliable", "propaganda"]
by_type = posts[flag_types].sum()
(by_type / by_type.sum()).round(3).sort_values(ascending=False).rename("share of flags").to_frame()""",
    ),
    (
        "code",
        """by_category = posts["subreddit_cat"].value_counts(normalize=True).round(3)
by_month = posts.groupby("month").size().rename("flagged posts")
print(by_category.to_string(), "\\n")
print(by_month.to_string())""",
    ),
    (
        "md",
        """### Table 5 - volume of data

Keyword matches are counted from the monthly MATCH files. The numbers of posts and comments
need the raw monthly files and are quoted from the paper.""",
    ),
    (
        "code",
        """table_5 = pd.DataFrame(
    {"N": [3_395_847, 17_254_621, len(matches), len(posts)]},
    index=["posts (from the paper)", "comments (from the paper)", "keyword matches",
           "POS matches"],
)
table_5["N/month"] = (table_5["N"] / 6).astype(int)
table_5["N/week"] = (table_5["N"] / 23).round(2)
table_5["N/day"] = (table_5["N"] / 182).round(2)
table_5""",
    ),
    (
        "md",
        """### Section 5.1 - weekly share of posts flagged (Figure 2)

The weeks with the highest share of flagged posts.""",
    ),
    (
        "code",
        """weekly = (
    pd.read_csv(DATA / "dashboard" / "app_subr_week.csv").query("week <= 23")
    .groupby("week")[["flag", "n_subm_kept"]].sum()
)
weekly["percent flagged"] = (100 * weekly["flag"] / weekly["n_subm_kept"]).round(1)
weekly.sort_values("percent flagged", ascending=False).head(5)""",
    ),
    (
        "md",
        """## 5. Section 5.4 - topics among COVID-19 posts

K-means with three clusters on the five title-embedding components, as in the dashboard's
Clustering tab. Cluster numbers are arbitrary and the solution depends on the random start.""",
    ),
    (
        "code",
        """from sklearn.cluster import MiniBatchKMeans

from stroo.dashboard.components import common_words
from stroo.dashboard.data import COMPONENTS

covid = posts[posts["subm_covid"]].copy()
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=777).fit(covid[COMPONENTS])
covid["cluster"] = kmeans.predict(covid[COMPONENTS]) + 1
for cluster, titles in covid.groupby("cluster")["subm_title"]:
    words = ", ".join(common_words(titles, n_words=9)["word"])
    print(f"CL{cluster} ({len(titles):,} posts): {words}")
share = pd.crosstab(covid["month"].astype(int), covid["cluster"], normalize="index").round(2)
share.loc[[1, 6]]""",
    ),
    (
        "md",
        """## 6. Section 5.5 - comparison with a fact-checking website

`SSD.csv` has one row per post shared from January to June 2020; its `fakedomain` column
marks posts that link to a website on NewsGuard's list.""",
    ),
    (
        "code",
        """ssd = pd.read_csv(DATA / "factcheck" / "SSD.csv")
result = pd.Series(factcheck.compare_with_suspicious(ssd))
result[[
    "posts from suspicious domains",
    "posts from suspicious domains flagged",
    "share of posts from suspicious domains flagged",
    "share of posts from other domains flagged",
    "share of suspicious domains flagged",
    "share of other domains flagged",
]].rename("value").to_frame()""",
    ),
    ("code", """factcheck.suspicious_domains_per_subreddit(ssd).head(6)"""),
]


def build() -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell(source) if kind == "md" else new_code_cell(source)
            for kind, source in CELLS
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NotebookClient(notebook, timeout=3600, resources={"metadata": {"path": str(REPO)}}).execute()
    OUTPUT.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
