"""Activate a licence from the environment, then assert the tier out loud.

Used by the sweep workflow. A sweep benchmark run without a licence does not
fail, it produces a wrong number: every unlicensed fan-out call waits out a
fixed interval before doing any work, so the stopwatch measures the wait. On a
100-cell grid that read 5.00 s against vectorbt's 0.17 s, which would publish
"vectorbt is 29x faster" from a run where the engine barely ran.

So this exits non-zero rather than let the benchmark continue unlicensed.

The key is written to disk by `activate`, so every child process the harness
spawns afterwards picks it up without seeing the secret itself.

    MANIFOLDBT_CI_LICENSE=<signed key> python ci_activate.py
"""
from __future__ import annotations

import os
import sys

import manifoldbt as mbt

# The American spelling, because that is how the repository secret is actually
# named. Both are accepted: the two spellings are one typo apart, and the cost
# of getting it wrong is a benchmark job that dies at the first step.
VARS = ("MANIFOLDBT_CI_LICENSE", "MANIFOLDBT_CI_LICENCE")


def main() -> int:
    key = ""
    for var in VARS:
        key = (os.environ.get(var) or "").strip()
        if key:
            break
    if not key:
        print(" and ".join(VARS) + " are empty or unset: "
              "refusing to benchmark sweeps unlicensed.")
        return 1

    # Never print the key or the activation message: the message carries the
    # licensee's address, and a public run log is not the place for it.
    try:
        mbt.activate(key)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        print(f"activation failed: {type(exc).__name__}")
        return 1

    _used, _limit, is_pro = mbt._native._combo_budget()
    if not is_pro:
        print("activation did not yield a licensed tier: refusing to continue.")
        return 1
    print("licensed tier active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
