"""Project paths and logging, in one place.

Every module takes its locations from here instead of hard-coding a home
directory. Two environment variables override the defaults:

``STROO_HOME``
    Project root (the folder holding ``config/``). Defaults to the repository
    this file lives in.
``STROO_DATA_DIR``
    Where data is read and written. Defaults to ``<STROO_HOME>/data``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("STROO_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = Path(os.environ.get("STROO_DATA_DIR", PROJECT_ROOT / "data"))

#: Monthly submission and comment files: ``SUBM_yyyy_mm.csv``, ``COMM_yyyy_mm.csv``.
RAW_DIR = DATA_DIR / "raw"
#: Matched sentences (``MATCH_yyyy_mm.csv``) and joined tables (``UNITED_*.csv``).
PROCESSED_DIR = DATA_DIR / "processed"
#: The six ``app_*.csv`` tables read by the dashboard.
DASHBOARD_DATA_DIR = DATA_DIR / "dashboard"
#: Manually annotated comments used for validation.
LABELLED_DIR = DATA_DIR / "labelled"
#: Universal Sentence Encoder weights (not distributed with the repository).
MODEL_DIR = DATA_DIR / "models" / "universal-sentence-encoder_4"
LOG_DIR = DATA_DIR / "logs"

COVID_KEYWORDS_FILE = CONFIG_DIR / "covid_keywords.txt"
SUBREDDITS_FILE = CONFIG_DIR / "subreddits.csv"

LOG_FORMAT = "%(asctime)s %(name)-14s [%(levelname)-8s] %(message)s"
LOG_DATE_FORMAT = "%y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a logger that writes to the console and, optionally, to ``LOG_DIR``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_DIR / log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
