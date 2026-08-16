# Community sweep gating — shipped design

Two mechanisms, both in `crates/bt-python/src/lib.rs` around
`require_combo_limit`:

1. a **cumulative combo budget** per process (256), and
2. a **wall-clock rate gate** — 5 s between Community sweeps, serialised
   machine-wide by an exclusive file lock.

Pro is untouched: the gate returns before both.

Measured effect on the bypass this exists to stop (one fresh interpreter per
slice, since the counter dies with the process):

| strategy          | combos | Pro    | bypass, sequential | bypass, parallel | friction |
|-------------------|--------|--------|--------------------|------------------|----------|
| RSI + EMA (light) | 5,000  | 5.9 s  | 103 s              | 95 s             | 16.2x    |
| RSI + EMA (light) | 20,000 | 23.3 s | 413 s              | 390 s            | 16.7x    |
| SMA bands (heavy) | 5,000  | 10.0 s | 107 s              | 95 s             | 9.5x     |
| SMA bands (heavy) | 20,000 | 40.0 s | 424 s              | 391 s            | 9.8x     |

Friction is stable *within* a strategy (16.2 → 16.7, 9.5 → 9.8) and differs
*between* them: it no longer depends on grid size at all, only on how expensive
the strategy is per combination. At the previous 500-combo cap the same four
cells read 7.7x / 8.7x / 5.1x / 5.2x — halving the cap roughly doubled them, as
the model predicts.

Every cell read **1.0x** before this work: slicing a grid across processes cost
the same as one big licensed call.

## The hole

The Community limit (`COMMUNITY_MAX_SWEEP_COMBOS`, 500 at the time) used to be
checked **per call**. A parameter grid is sliceable, so splitting it bypassed
the cap:

```python
for i in range(0, 500, 5):                       # 100 calls x 500 combos
    bt.run_sweep_lite(strat, {"dev_up": DU[i:i+5], "dev_dn": DD}, cf, store)
```

| run                                | time   | per combo |
|------------------------------------|--------|-----------|
| 1 call x 50,000 combos (Pro)       | 89.9 s | 1.80 ms   |
| 100 calls x 500 combos (Community) | 90.3 s | 1.81 ms   |

The split cost **+0.5%**, and per-call overhead is ~10 ms once data is cached,
so no per-call charge could make slicing expensive without hitting legitimate
small sweeps first.

## 1. Cumulative combo budget

A process-lifetime `AtomicU64`. Non-Pro calls are refused when
`total + n_combos > COMMUNITY_MAX_SWEEP_COMBOS`; otherwise the total is
CAS-incremented — a plain load+store would let two concurrent sweeps both slip
under the cap. A refused call consumes no budget and pays no wait. Pro skips
check and accounting. `_native._combo_budget()` exposes `(used, limit, is_pro)`
read-only.

All native fan-out entry points share one total: `run_sweep`, `run_sweep_lite`,
`run_batch`, `run_batch_lite`, `run_sweep_2d`, `run_stability`. A single
`bt.run()` is ungated.

The Python-side `_require_pro_over_combos` stays a fast-fail courtesy check for
the single-call case; the authoritative gate is native, since the Python layer
is trivially patchable.

This alone leaves the restart bypass wide open — the counter dies with the
process — which is what mechanism 2 prices.

## 2. Rate gate: 5 s, serialised by a file lock

Before each accepted Community call, the gate waits `SWEEP_MIN_INTERVAL` (5 s)
while holding an exclusive lock on `{license_dir}/sweep.lock`.

**The lock is the mechanism, not plumbing.** Waiting costs no cores, so N
concurrent processes would otherwise all clear the same 5 s — the way every
sleep-based limiter dies. Serialising the wait behind one machine-wide lock
makes N slices cost N x 5 s whatever the core count. The measurements confirm
it exactly: 19 slices in parallel took 95 s and 78 took 390 s, against a
theoretical floor of 95 s and 390 s. Parallelising buys nothing.

**Stateless on purpose.** An earlier shape recorded "last sweep at T" and only
waited the remainder — friendlier, since an idle user pays nothing. But the
record is the attack surface: restoring a copy from a minute ago grants an
immediate pass, and a MAC prevents forgery, not replay (a *stale* restored file
is the permissive one, so replay always favours the attacker). Here the file
carries no data, only the lock. Nothing to roll back; deleting it merely makes
the next caller recreate it, and the wait still happens.

**Fails closed on an unwritable state directory** (read-only HOME, locked-down
CI): the wait still happens, only without cross-process serialisation. Failing
open would make an unwritable HOME the bypass.

**Paid up front**, before results exist, so killing the process to dodge the
wait also discards the run. The GIL is released throughout.

Implemented with `std::fs::File::lock` — stable since Rust 1.89, so no new
dependency, but that is the minimum toolchain for this crate now.

### Why not a CPU penalty (what this replaced)

The previous mechanism burned ~3 s of counted CPU per process. It worked
(3.7x–6.2x) but had three defects the rate gate does not:

- **Cost in core-seconds, not seconds.** Friction was
  `1 + fixed_cost / (Pro cost of the slice)`, so the same penalty bought 6.2x on
  a light strategy, 3.9x on a heavy one, and would fall to ~1.2x on 3 years of
  bars. The rate gate's floor is wall-clock, so strategy weight and dataset size
  drop out.
- **Machine-dependent**, needing a runtime calibration and per-thread clamps to
  stay near 3 s across CPUs. The rate gate needs neither.
- **Wasteful and visible.** Burning every core for 3 s is indefensible if a
  user profiles it. Waiting costs nothing and leaves the machine usable.

## Tuning

Behaviour is now predictable:

```
friction ≈ (combos / cap) x 5 s / (Pro time for those combos)
```

The cap is therefore the lever, and its effect is calculable rather than
measured: halving it roughly doubles the friction, at the cost of a smaller free
allowance. Verified — moving from 500 to 256 took the four cells above from
5.1x–8.7x to 9.5x–16.7x, a 1.9x shift against a predicted 1.95x.

Current settings: `COMMUNITY_MAX_SWEEP_COMBOS = 256`, `SWEEP_MIN_INTERVAL = 5 s`.
A legitimate free user pays 5 s per sweep, capped at 256 combinations per
process — still a 16x16 grid.

## Rejected alternatives

- **Disk-persisted counter, or a required "ticket" file.** Deleted with `rm`,
  or restored with `cp`; a hash prevents forgery, not replay.
- **Shared memory / sentinel process.** One `pkill` away, and a library
  spawning hidden background processes is an antivirus profile on Windows.
- **Delay measured from a recorded last-sweep time.** Needs history, which is
  restorable — see above. The stateless always-wait shape avoids it.
- **Plain sleep without the lock.** Overlaps for free across processes; this is
  the single reason the lock exists.
- **Penalty scaled by combos, or by combos x dataset span.** Measured and
  dropped: a legitimate 500-combo sweep reached 3 s and 4.4 s respectively.
- **`run_sweep_lite` restricted to Pro.** Would work — capability cannot be
  sliced around — but turned down on positioning: the free tier should be
  capped in *how much*, not degraded in *how well*.

## Known limits

- **The clock is the user's.** `faketime`/`LD_PRELOAD` can shorten the sleep,
  and per-slice mount namespaces (`unshare`) give each process its own lock
  file. Both are deliberate technical acts, unlike "restart Python".
- **A patched wheel defeats any client-side check.** This is the ceiling of the
  whole approach; the goal is to make bypassing cost more than a subscription,
  not to make it impossible.
- Escalation path if telemetry ever shows this is not enough:
  `docs/sweep-key-rate-limit-plan.md`.

## Related findings (tracked separately)

1. `run_sweep` (full Arrow output) is OOM-killed around ~6.8 MB/combo: 2,000
   combos ≈ 13.4 GB RSS, 3,000 → SIGKILL, no Python exception, no warning. Only
   Pro users can reach it, since the Community cap keeps grids below the danger
   zone — a paying-customer-only failure. Needs an upfront memory estimate and
   a clean error pointing at `run_sweep_lite`.
2. Level-fill via `execution_source` regenerates the execution bar feed per
   combination (~80x slower sweeps); an expression-based execution price would
   move the level back into parameter space. Separate design.

## Measurement note

Per-combo cost is flat across grid size — 1.21 / 1.13 / 1.11 / 1.17 / 1.11
ms/combo at 500 / 2k / 10k / 50k / 200k on one strategy. An earlier revision of
this document claimed the big call degraded at scale; that was wrong, and came
from comparing grids with different *content*. Per-combo cost tracks trade
count, not grid size — which is also why friction varies by strategy.
