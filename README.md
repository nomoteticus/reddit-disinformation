# reddit-disinformation

Code for the data processing pipeline and dashboard behind:

> Achimescu, V., & Chachev, P. D. (2021). **Raising the Flag: Monitoring User Perceived Disinformation on Reddit.** *Information*, 12(1), 4. https://doi.org/10.3390/info12010004

The paper uses *informal flags* — Reddit comments such as "this is fake news" or "not a reliable source" — as a signal of which posts users perceive as false information. A rule-based matcher (keywords + part-of-speech tags + dependency parsing, built with spaCy) finds these flags in millions of comments, and the flagged posts are shown in an interactive Plotly/Dash dashboard ("STROO?", short for "Is it true?"). An extended version of the paper is Chapter 4 of my PhD dissertation (University of Mannheim, 2021).

## Pipeline

The numbered folders run in order. Each step reads from and writes to monthly CSV files in an `output/` folder (not in this repository; see *Data* below), so the pipeline could be re-run daily and only process what is new.

```
1_scraping  →  2_findflags  →  3_embed  →  dashboard
 (collect)      (detect)       (cluster)    (explore)
                    ↓
                4_check, 6_disinfo (validate)
```

| Folder | What it does |
|---|---|
| `input/` | Configuration: COVID-19 search keywords, the list of tracked subreddits with their manual classification (`subr_classification.csv`: category, language, keep/drop), and the codebook used for manual annotation. |
| `1_scraping/` | Data collection from Reddit through the Pushshift API (`psaw`). `01_*` scripts discover subreddits that share news links about COVID-19, tabulate their daily activity and metadata (subscribers, comments per post, share of removed posts), and support the manual classification that decides which subreddits to track. `02_*` and `03_*` back-fill past submissions and comments; `04_scr_submcomm_realtime.py` is the daily job that appends new submissions (`SUBM_yyyy_mm.csv`) and comments (`COMM_yyyy_mm.csv`), with retries, timeouts and logging. `func_submcomm.py` holds the shared helpers; `praw/` contains earlier experiments with the official Reddit API. |
| `2_findflags/` | Flag detection. `all_subreddits/functions_pos_match.py` defines the vocabulary (six flag types: disinformation/misinformation, fake news, misleading/clickbait, unreliable, propaganda, bullshit), the 21 syntactic patterns of the POS matcher, and the regular expressions that filter out sarcasm, bots, and posts that are themselves about fake news; `test_func_pos_match.py` tests the patterns on example sentences. `21_sent_pos_df.py` pre-filters comments by keyword, parses them with spaCy, and writes every matching sentence to `MATCH_yyyy_mm.csv`. `22_bring_together.py` joins matches, comments, and submissions and builds the weekly aggregates the dashboard reads (by flag, submission, subreddit, author, and linked web domain). `news_subreddits/` holds the first prototype on a smaller set of news subreddits. |
| `3_embed/` | Computes Universal Sentence Encoder embeddings of post titles and reduces them with PCA. The dashboard uses these for "similar posts" and for k-means topic clustering. The notebooks check flag frequencies by pattern and type. |
| `4_check/` | Validation of the POS matcher. `41_*` scripts draw stratified samples of comments for manual annotation by two coders; `42_*` scripts train baseline machine-learning classifiers (TF-IDF + random forest, scikit-learn) on the annotated comments and compare them with the rule-based matcher (precision, recall, F1) on a later, unseen test period. `july/` holds the first annotation round. |
| `6_disinfo/` | Notebooks comparing the web domains flagged by Reddit users with an external list of unreliable news sources compiled by fact-checkers. |
| `dashboard/` | The Plotly/Dash app. `APP_corona_v0xx.py` files are successive versions; `func_dashboard.py` holds the data-preparation and plotting functions and `assets/` the CSS and images. `onserver/` is the deployed version (Flask app served with Gunicorn and Nginx on an Ubuntu VM: `wsgi.py`, `runapp.py`, version history). Left panel: filters for period, flag type, minimum number of flags, subreddit, and source reliability. Tabs: highlights, flagged posts (with similar-post search), topic clustering, and method. |
| `analysis/` | Exploratory notebooks and R scripts that go beyond the published paper (comment embeddings, similarity between subreddits, moderator deletions). Work in progress; not needed to reproduce the paper. |
| `art/` | Logo files for the dashboard. |

## Data

The raw and processed data are too large for GitHub. The datasets, annotated samples, and a snapshot of the dashboard source used in the paper are archived on Figshare:

- Data collection and analysis: https://doi.org/10.6084/m9.figshare.13174145.v1
- Dashboard of flagged Reddit posts: https://doi.org/10.6084/m9.figshare.13174136.v1

## Status

This is research code from 2020–2021, kept as it was when the paper and the dissertation were written. The dashboard was publicly hosted and updated daily during that period and is no longer online. The scraping scripts rely on the public Pushshift API, which has since been restricted, so they document the method rather than run as-is. Paths are set through a `cwd.txt` file or the repository root.

Main tools: Python 3.6, pandas, spaCy, scikit-learn, TensorFlow Hub (Universal Sentence Encoder), Plotly/Dash, Flask, Gunicorn, Nginx; R (tidyverse) for some analyses.

## Citation

If you use this code or the method, please cite the paper above.
