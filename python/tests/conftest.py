import os

import pytest

_CRATE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_ROOT = os.path.join(
    _CRATE_ROOT, "..", "bt-core", "tests", "fixtures", "golden",
)


@pytest.fixture
def golden_buy_hold_dir():
    """Path to the buy_and_hold golden fixture directory.

    Skips instead of failing when the directory is absent: the 1-minute fixtures
    are part of the engine's own test data and are not distributed with the
    package, so a clone that only has the Python suite should report "not
    applicable" rather than an error it cannot act on. The golden test that runs
    anywhere is `test_golden_daily.py`, whose fixtures live next to it.
    """
    path = os.path.join(GOLDEN_ROOT, "buy_and_hold", "v1")
    if not os.path.isdir(path):
        pytest.skip("1-minute golden fixtures are not distributed with the package")
    return path
