import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "legacy"))

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def legacy():
    """The original 2020 matcher module (see tests/legacy/legacy_port.py)."""
    from legacy_port import load_legacy

    return load_legacy()


@pytest.fixture(scope="session")
def matcher():
    from stroo.flags import get_matcher

    return get_matcher()


@pytest.fixture(scope="session")
def data_dir():
    """The local, git-ignored data folder; tests needing it are skipped without it."""
    if not (DATA / "baseline").exists():
        pytest.skip("local data/ folder not available")
    return DATA
