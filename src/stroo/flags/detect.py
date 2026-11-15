"""Find informal flags in the monthly comment files.

For every ``COMM_yyyy_mm.csv`` in ``data/raw`` this writes a
``MATCH_yyyy_mm.csv`` to ``data/processed`` with one row per sentence that
contains a flag-type keyword, and indicator columns telling whether the
sentence matches a pattern (``flag``) and which keywords it contains.

Steps, per comment (Section 4 of the paper):

1. keyword filter - keep comments containing a flag-related word;
2. sarcasm filter - drop comments with ``/s``, scare quotes or "jk";
3. split into sentences and parse each with spaCy;
4. apply the patterns to every sentence.

The job is incremental: when MATCH files already exist only the two most
recent months are revisited, and comments already present in a MATCH file are
not parsed again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import settings
from ..io import write_csv
from .filters import KEYWORD_REGEX, SARCASM_REGEX
from .matcher import LABELS, FlagMatcher, get_matcher

MATCH_COLUMNS = ["comm_id", "sent_id", "sent"] + LABELS
COMMENT_COLUMNS = ["subreddit", "link_id", "id", "parent_id", "body"]
CHUNK_SIZE = 100_000

log = settings.get_logger("stroo.detect", "pos_embed_unite.log")


def monthly_files(folder: Path, prefix: str) -> list[str]:
    """Sorted names of the ``<prefix>_yyyy_mm.csv`` files in ``folder``."""
    if not folder.exists():
        return []
    pattern = re.compile(rf"{prefix}_20\d\d_\d\d\.csv")
    return sorted(path.name for path in folder.iterdir() if pattern.fullmatch(path.name))


def prefilter_comments(comments: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the comment text and apply the keyword and sarcasm filters."""
    comments = comments.assign(body=comments["body"].astype(str).str.lower())
    comments = comments[comments["body"].str.contains(KEYWORD_REGEX)]
    return comments[~comments["body"].str.contains(SARCASM_REGEX)]


def match_comments(
    comment_ids, bodies, matcher: FlagMatcher | None = None, n_process: int = 1
) -> pd.DataFrame:
    """Parse comments and return one row per sentence with a flag-type keyword.

    ``bodies`` are expected lower-cased (see ``prefilter_comments``).
    """
    matcher = matcher or get_matcher()
    docs = matcher.nlp.pipe(bodies, n_process=n_process, disable=["ner"])
    rows = []
    # No `strict=`: this module also runs on Python 3.8 (environment-paper.yml).
    for comment_id, doc in zip(comment_ids, docs):  # noqa: B905
        for sentence_id, sentence in enumerate(doc.sents):
            labels = matcher.labels(sentence)
            if labels:
                row = {"comm_id": comment_id, "sent_id": sentence_id, "sent": sentence.text}
                row.update(dict.fromkeys(labels, 1))
                rows.append(row)
    matches = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    matches[LABELS] = matches[LABELS].fillna(0).astype("int64")
    return matches


def update_match_file(
    comment_file: Path,
    match_file: Path,
    matcher: FlagMatcher | None = None,
    n_process: int = 1,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Add the flags of comments not yet in ``match_file``; return sentences added."""
    if match_file.exists():
        matches = pd.read_csv(match_file)
    else:
        matches = pd.DataFrame(columns=MATCH_COLUMNS)
        write_csv(matches, match_file)
    already_matched = set(matches["comm_id"])

    n_added = 0
    chunks = pd.read_csv(comment_file, chunksize=chunk_size, usecols=COMMENT_COLUMNS)
    for chunk_number, chunk in enumerate(chunks, start=1):
        comments = prefilter_comments(chunk)
        comments = comments[~comments["id"].isin(already_matched)]
        log.debug("Trying %d comments", len(comments))
        if comments.empty:
            # Nothing new in this chunk: stop scanning this file.
            log.warning("No new comments in comm file: %s", comment_file.name)
            break

        new_matches = match_comments(comments["id"], comments["body"], matcher, n_process)
        if new_matches.empty:
            log.warning("%5dK. No new sentences.", chunk_number * chunk_size // 1000)
            break

        n_added += len(new_matches)
        matches = pd.concat([new_matches, matches], ignore_index=True)
        log.debug(
            "%5dK. Done: %4d comments, %4d sentences. Flags: %s",
            chunk_number * chunk_size // 1000,
            new_matches["comm_id"].nunique(),
            len(new_matches),
            new_matches["flag"].sum(),
        )
        write_csv(matches, match_file)
        log.info("Saved match file: %s", match_file.name)
    return n_added


def run(raw_dir: Path | None = None, processed_dir: Path | None = None, n_process: int = 1) -> None:
    """Update the MATCH files for the comment files that may have changed."""
    raw_dir = raw_dir or settings.RAW_DIR
    processed_dir = processed_dir or settings.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    comment_files = monthly_files(raw_dir, "COMM")
    match_files = monthly_files(processed_dir, "MATCH")
    log.info("Started pattern search")
    log.debug("COMM files: %s", comment_files)
    log.debug("MATCH files: %s", match_files)

    # First run: every month. Later runs: only the two most recent months.
    to_process = comment_files[-2:] if match_files else comment_files
    matcher = get_matcher()
    for comment_file in to_process:
        log.info("Started comm file: %s", comment_file)
        update_match_file(
            raw_dir / comment_file,
            processed_dir / comment_file.replace("COMM", "MATCH"),
            matcher,
            n_process,
        )
        log.info("Finished comm file: %s", comment_file)
