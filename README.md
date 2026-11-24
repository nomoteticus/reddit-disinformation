# reddit-disinformation

Code for the data pipeline and dashboard behind:

> Achimescu, V., & Chachev, P. D. (2021). **Raising the Flag: Monitoring User Perceived Disinformation on Reddit.** *Information*, 12(1), 4. https://doi.org/10.3390/info12010004

The paper uses *informal flags* — Reddit comments such as "this is fake news" or "not a reliable source" — as a signal of which posts users perceive as false information. A rule-based matcher (keywords + part-of-speech tags + dependency parsing, built with spaCy) finds these flags in millions of comments, and the flagged posts are shown in an interactive Plotly/Dash dashboard, **STROO?** (short for "Is it true?"). An extended version of the paper is Chapter 4 of my PhD dissertation (University of Mannheim, 2021).

```
collect  →  detect  →  aggregate  →  dashboard
(Reddit)    (flags)    (tables +     (explore)
                        embeddings)
                ↓
            validation (manual labels, ML baselines, fact-checker list)
```

## Quick start

```bash
conda env create -f environment.yml
```

```bash
conda activate stroo
```

```bash
python -m spacy download en_core_web_sm
```

Download `app_flags_day.csv`, `app_subm_day.csv` and `app_subr_week.csv` from [Figshare](https://doi.org/10.6084/m9.figshare.13174136) into `data/dashboard/`, then:

```bash
stroo dashboard
```

and open http://127.0.0.1:8050.

## Repository layout

| Path | Contents |
|---|---|
| `src/stroo/` | The Python package (see below). |
| `config/` | COVID-19 search keywords, the tracked subreddits with their manual classification (`subreddits.csv`), the codebook of the joined table. |
| `notebooks/paper_results.ipynb` | The paper's tables and figures recomputed from the published data, next to the published values. |
| `scripts/` | `check_paper_reproduction.py` (exact reproduction of the 2020 matching) and `build_paper_notebook.py`. |
| `tests/` | Unit tests, and tests that compare the code with the original scripts and with the paper. |
| `deploy/` | Gunicorn/systemd/Nginx files and a deployment guide for the dashboard. |
| `docs/reference/` | Pattern frequencies and the most frequent matched sentences, July 2020. |

### The package

| Module | What it does | Paper |
|---|---|---|
| `stroo.collect` | Selects subreddits that share news links about COVID-19; daily scrape of their posts and of the comments on link posts, via the Pushshift API. | Section 3 |
| `stroo.flags` | The vocabulary (`vocabulary.py`), the 21 syntactic patterns (`patterns.py`), the matcher, the keyword/sarcasm/bot filters, and the job that writes matched sentences to `MATCH_yyyy_mm.csv`. | Section 4, Appendix B |
| `stroo.embed` | Universal Sentence Encoder embeddings of post titles, reduced with PCA. | Section 7.3 |
| `stroo.aggregate` | Joins flags, comments and posts; builds the tables read by the dashboard. | Appendix A |
| `stroo.validation` | Sampling of comments for manual annotation, precision/recall of the matcher, machine-learning baselines, comparison with a fact-checker's list. | Sections 4.4, 5.5 |
| `stroo.dashboard` | The Dash app. | Section 7 |

Paths are set in `stroo/settings.py`. By default data is read from and written to `data/` in the repository; set `STROO_DATA_DIR` to use another folder.

## Running the pipeline

```bash
stroo collect
```

```bash
stroo detect
```

```bash
stroo aggregate 2020
```

Each step reads and writes monthly CSV files, so it can run daily and only process what is new:

| Step | Reads | Writes |
|---|---|---|
| `collect` | Pushshift API | `data/raw/SUBM_yyyy_mm.csv`, `COMM_yyyy_mm.csv` |
| `detect` | `COMM_*.csv` | `data/processed/MATCH_yyyy_mm.csv` |
| `aggregate` | `SUBM_*`, `COMM_*`, `MATCH_*` | `data/processed/UNITED_yyyy_*.csv`, `data/dashboard/app_*.csv` |

The same functions can be used from Python:

```python
from stroo.flags import get_matcher

matcher = get_matcher()
matcher.comment_is_flag("This is fake news.")                # True
matcher.comment_is_flag("Fake news is an overused trope.")   # False
```

### Collecting new data

`stroo collect` uses the public Pushshift API through `psaw`. Pushshift closed that API in 2023, so the collection code documents how the data of the paper were gathered but cannot run as it is. A replacement data source has not been added yet.

### Embeddings

`stroo aggregate` needs the Universal Sentence Encoder (version 4) and TensorFlow (`pip install -e ".[embed]"`). Download the model from https://tfhub.dev/google/universal-sentence-encoder/4 and unpack it into `data/models/universal-sentence-encoder_4/`.

## The dashboard

Left panel: filters for the period, the type and number of flags, the subreddit, and COVID-19 related posts. Tabs:

| Tab | Shows |
|---|---|
| Highlights | Headline numbers, trend of flagged posts, latest flagged posts, frequent title words. |
| Flags | Most common and most recent flag sentences. |
| Submissions | Flagged posts with their flags, removal status and similar posts. |
| Subreddits | Treemap of flags per subreddit; weekly trend per subreddit category, as counts or percent. |
| Clustering | K-means topics of post titles, with keywords and the posts of each cluster. |
| Domains | Websites linked by the flagged posts. |
| Method | A short description of the method. |

The dashboard is an archive of the 2020 data, not a live monitor. It shows no user names: the author tables are not loaded, user mentions in comments and titles are masked, and posts removed by moderators or deleted by their authors are left out (set `STROO_SHOW_REMOVED=1` to include them for research use).

The app is the 2021 redesign of the dashboard (v0.1.2) with tabs of the version described in the paper added back. To put it on a server, see [`deploy/README.md`](deploy/README.md).

## Data

The raw and processed data are too large for GitHub and contain user names and comment text. Archived on Figshare:

- Dashboard tables and source snapshot: https://doi.org/10.6084/m9.figshare.13174136
- Descriptive analyses: https://doi.org/10.6084/m9.figshare.13174145
- Manual annotations: https://doi.org/10.6084/m9.figshare.13315259

## Reproducing the paper

| What | How | Notes |
|---|---|---|
| Flag matching | `environment-paper.yml` + `scripts/check_paper_reproduction.py` | Identical to the 2020 output with spaCy 2.3.5. Newer spaCy models parse some sentences differently, so current versions give close but not identical results. |
| Tables 1 to 5, Sections 5 and 5.5 | `notebooks/paper_results.ipynb` | Recomputed from the published data with the package functions. The machine-learning baselines of Table 4 depend on library versions. |

## Tests

```bash
pytest
```

Tests marked `data` compare the code with the original scripts and with the paper. They need the local `data/` folder and are skipped without it.

## History

The code was written in 2020-2021 and restructured in 2026 into the package described here. The original scripts, including every earlier version of the dashboard and the exploratory analyses that are not part of the paper, are kept under the git tag `archive/pre-cleanup`.

## Citation

If you use this code or the method, please cite the paper above.

## Licence

The code is released under the [MIT Licence](LICENSE). The data on Figshare and the article have their own licences.
