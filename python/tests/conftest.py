import os

import pytest

_CRATE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_ROOT = os.path.join(
    _CRATE_ROOT, "..", "bt-core", "tests", "fixtures", "golden",
)


@pytest.fixture
def golden_buy_hold_dir():
    """Path to the buy_and_hold golden fixture directory."""
    path = os.path.join(GOLDEN_ROOT, "buy_and_hold", "v1")
    assert os.path.isdir(path), f"golden fixture dir not found: {path}"
    return path
