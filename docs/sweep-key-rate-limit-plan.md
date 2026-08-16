# Server-issued sweep keys — escalation plan

Status: **specified, deliberately not built.** The shipped local rate gate
(`docs/sweep-combo-limit-plan.md`) measures 9.5x–16.7x at a 256-combo cap, and
lowering the cap further scales it linearly — the level this design was meant to
buy,
without a server, without making the free tier online-only, and without a
privacy review.

Build this **only** if telemetry shows the local gate being defeated at scale.
What would justify it: evidence of `faketime`/`LD_PRELOAD` clock manipulation
or per-slice namespaces in the wild, i.e. the two attacks the local gate cannot
answer. Nothing else here is worth the cost.

The rest of the document is the spec, kept complete so the decision can be
revisited without redoing the analysis.

## 1. What it does

A sweep must present a valid, unexpired, server-signed key. The key carries a
**budget of combinations**, not a single authorisation, so a burst of small
sweeps shares one round-trip.

| tier      | rate            | combos per key | TTL   | round-trips           |
|-----------|-----------------|----------------|-------|-----------------------|
| Community | 1 key per 5 s   | 100            | 30 s  | 1 per 100 combos      |
| Pro       | none            | unlimited      | 24 h  | 1 per day             |

Expected effect, from the measured Pro baselines (3 months of 1m bars, 4 cores):

| combos | Pro    | Community floor     | friction |
|--------|--------|---------------------|----------|
| 5,000  | 8.0 s  | 50 keys → 4.2 min   | ~31x     |
| 20,000 | 26.6 s | 200 keys → 16.7 min | ~38x     |

These hold on any machine, strategy and dataset size — but so does the shipped
local rate gate, for the same reason (a wall-clock floor rather than CPU work).
The server's only remaining advantage is that its clock is out of the user's
reach: no `faketime`, no namespace trick. That is the entire delta now.

## 2. Why the server, and why no offline lane

Every client-side variant was built or analysed and fails the same two ways
(measurements in the companion doc):

- **State dies with the process.** The cumulative counter resets on restart. On
  disk it is one `rm` away; as a required "ticket" file it is one `cp` away —
  a MAC prevents forgery, not replay, and a *stale* restored file is more
  permissive, so replay always favours the attacker.
- **A fixed client *CPU* cost buys a variable ratio.** Friction was
  `1 + fixed_cost / (Pro cost of the slice)`, so the CPU penalty gave 6.2x on a
  light strategy and 3.9x on a heavy one. (The lock-serialised wall-clock gate
  that replaced it does not have this defect — it is why this plan is no longer
  urgent.)

**Why no offline lane.** "Free users get N combos offline, the key is only
needed beyond that" sounds like a friendly compromise and is worth nothing: the
offline allowance resets on restart, so an abuser stays in the offline lane
forever and never requests a key. Any tolerated offline mode is a hole exactly
the size of the allowance. The choice is binary — accept the network
requirement, or keep the local gate's 9.5x–16.7x.

## 3. Protocol

```
client                                      server
  |-- POST /api/sweep-key ---------------->|  { device_hash, license_hash?,
  |                                        |    version, requested_combos }
  |                                        |  identify -> rate + quota check
  |<---------- signed key -----------------|  { combos, expires_at, device_hash,
  |                                        |    nonce, tier, sig }
  |  verify sig (embedded public key)      |
  |  spend from the budget as sweeps run   |
```

**Key payload** (JSON, serialised canonically — reuse the existing licence rule
that payloads re-serialise byte-identically so signatures verify):

```json
{ "combos": 100, "expires_at": "2026-08-11T09:31:04.000Z",
  "device_hash": "…", "nonce": "…", "tier": "community" }
```

Signed Ed25519, verified against a public key compiled into the wheel. Reuse
`bt-license`: `ed25519-dalek` is already a dependency, `license.rs` already
verifies this exact shape, and `PUBLIC_KEY_BYTES` is already stored XOR-masked
and split rather than as a contiguous blob. **Use a separate key pair from the
licence one** so a leaked sweep-signing key cannot mint licences.

`device_hash` is inside the signature, so a key copied to another machine is
rejected locally, with no round-trip.

### Why the two TTLs differ

The 5 s rhythm is **not** enforced by the TTL — the server refuses to mint
before 5 s have passed, which is server state. The short Community TTL exists
to prevent **stockpiling**: with a one-hour TTL an abuser requests a key every
5 s for an hour, banks 720 keys, then burns them in parallel and gets 72,000
combinations at once. A 30 s TTL makes an unused key worthless while still
covering a burst of small sweeps.

Pro's 24 h TTL is the opposite trade: one round-trip, then a full day offline.
Its exposure is key sharing, already covered by the device binding in the
signature plus the existing per-licence device limit.

### Rate limiting caps throughput, not volume

At one key per 5 s a patient Community user still reaches ~72,000 combinations
per hour. If the intent is a ceiling rather than a drip, the server must also
hold a **daily total per account**. Recommended: yes, with a generous limit
(e.g. 5,000/day) — it costs one counter and turns "slow" into "bounded".

## 4. Client implementation

All in `crates/bt-license` (transport, crypto, state) and
`crates/bt-python/src/lib.rs` (the gate).

**4.1 — `bt-license`: new `sweep_key` module**

- `pub fn acquire(requested: u32) -> Result<SweepKey, LicenseError>`
  - Returns the process-cached key if unexpired with budget left.
  - Otherwise `POST /api/sweep-key`, verify signature, verify `device_hash`
    matches `device::device_hash()`, cache, return.
- **Cache in memory only, never on disk.** An on-disk key is a stockpile, and
  the TTL is short enough that persistence buys nothing.
- Reuse the telemetry HTTP pattern (`reqwest` + a current-thread tokio runtime),
  but **blocking** with a **500 ms timeout** — a library that freezes on its
  first call gets uninstalled. Distinguish the three failures explicitly:
  network unreachable, rate-limited (HTTP 429 + `retry_after`), quota exhausted
  (HTTP 402).
- Clock skew: trust `expires_at` against local time but also keep a local
  monotonic deadline from receipt; use whichever expires first, so a client
  clock set backwards cannot extend a key.

**4.2 — `bt-python`: replace the gate**

`require_combo_limit(py, n_combos, label)` becomes `require_sweep_budget`:

1. Pro (`check_feature("sweep")`) → acquire/refresh the 24 h key, return.
2. Community → `sweep_key::acquire(n_combos)`; on success debit the budget, on
   failure raise `PermissionError` with the server's reason.
3. Keep the in-process cumulative counter as a **local fast-fail** so an
   over-budget call fails without a round-trip.
4. Release the GIL around the request (`py.allow_threads`).

Call sites are unchanged: `run_sweep`, `run_sweep_lite`, `run_batch`,
`run_batch_lite`, `run_sweep_2d`, `run_stability`. A single `bt.run()` stays
ungated.

**4.3 — Error messages.** Three distinct, actionable texts:

- rate-limited → `"Community sweeps are limited to one every 5 s (4 s
  remaining). Upgrade at …"`
- daily quota → `"Community daily sweep quota reached (5,000 combinations).
  Resets at 00:00 UTC. Upgrade at …"`
- offline → `"Sweeps require a connection to www.manifoldbt.com on the
  Community tier. bt.run() and single backtests work offline."`

**4.4 — Remove** `community_rate_gate`, `SWEEP_MIN_INTERVAL` and the lock file.
The server rhythm replaces the local one; running both would charge the wait
twice.

## 5. Server implementation

- `POST /api/sweep-key` — authenticate (licence hash when present, device hash
  otherwise), enforce rate + daily quota, sign, return. Reject unknown or
  revoked licences.
- Storage: per-device `last_issued_at` and per-account `combos_today`. A single
  small table; the rate check is one read + one write.
- **Rate-limit by IP as well as by device.** Rotating `device_hash` to mint a
  fresh identity is the one bypass this design does not close; per-IP limits
  and requiring an account for Community keys raise the bar substantially.
- Metrics to emit from day one: keys issued, refusals by reason, distinct
  devices per account, combos per account per day. This is also the first real
  usage data on the free tier.
- Key rotation: publish the sweep-signing public key with a version byte in the
  payload so a future rotation does not break older wheels.

## 6. Rollout

1. **Ship telemetry first.** Add sweep-usage reporting to the current wheel and
   watch for a release. If the data shows nobody is chaining sweeps, stop here
   and keep the client-side gating — this design is not free.
2. **Server first, enforcement later.** Deploy the endpoint and let the client
   request keys *without* enforcing, logging refusals it would have made.
   Confirms capacity and false-positive rate against real usage.
3. **Enforce behind a version gate.** Only wheels ≥ the release that ships this
   require a key; older wheels keep working, so the change cannot brick an
   existing install.
4. **Announce before enforcing.** Community losing offline sweeps is a
   user-visible regression and must be in the release notes, not discovered.

## 7. Non-technical prerequisites

- **Privacy.** Unlicensed installs will contact the server for the first time —
  the current telemetry ping only fires when a licence is present
  (`guard.rs: if let Some(ref lic) = license`). Needs a privacy notice, a
  documented retention policy, and a GDPR review. `device_hash` is a salted
  hash of a machine ID: pseudonymous, not anonymous.
- **Availability becomes a product dependency.** If the endpoint is down, no
  Community user can sweep. Needs an uptime target, and a documented
  fallback decision (fail-closed is what makes the design hold; a fail-open
  switch is a bypass anyone can trigger by blocking a domain).

## 8. Attack surface after this lands

| attack                        | cost after |
|-------------------------------|-----------|
| loop inside one process       | blocked by the key budget |
| fresh process per slice       | useless — the server clock does not reset |
| copy the key file             | no key file exists; in-memory only |
| replay a captured key         | 30 s TTL, bound to the device |
| forge a key                   | Ed25519 |
| stockpile keys                | TTL shorter than the accumulation window |
| parallelise                   | useless — the rate is per account, not per process |
| rotate `device_hash`          | **works** — mitigated by per-IP limits and account requirement, not closed |
| block the network             | fails closed: no sweep |
| patch the wheel               | **works** — the ceiling of any client-side check |

The last two lines are the honest limit: this makes bypassing a deliberate,
technical act rather than a side effect of restarting Python.

## 9. Tests

- Community: two sweeps back to back — the second refused until 5 s elapse;
  101 combinations in one call refused outright; a burst of five 20-combo
  sweeps consumes one key, not five.
- Key expiry: a key held past its TTL is refused and transparently renewed.
- Signature: edited payload rejected; valid key minted for another
  `device_hash` rejected.
- Clock: local clock moved backwards does not extend a key (monotonic deadline
  wins).
- Pro: one key covers a 200,000-combo sweep; no rate limit; no penalty.
- Server unreachable: fails within the timeout with the offline message, never
  hangs.
- Restart: a fresh process cannot obtain a key faster than the server rhythm —
  the end-to-end property the whole design exists for. Same harness as the
  400-slice bypass benchmark used for the CPU penalty.

## 10. Still to decide

1. Daily quota: on or off, and at what value.
2. Rate: 5 s is a guess. It sets the free tier's usable throughput; pick it
   from the telemetry in step 6.1 rather than from intuition.
3. Whether Community keys require a registered account (email) or just a
   device hash. Accounts make per-IP limits far more effective, at the cost of
   a signup wall in front of a free tier.
