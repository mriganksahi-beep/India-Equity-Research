# moonshot_compounding — Build Log: Stages 4–6 (PIT flag + features)

**Scope of this log:** the chronological narrative and reasoning for Stages 4
through 6 — daily PIT membership flag, the remediation of the raw price panel
(Stage 3B), the seven derived features, and the final joined deliverable +
manifest. For the fast-lookup "things you must know" briefing, see
`IMPLEMENTER_MEMORY.md`. For Stages 1–3 (universe + raw price panel), see the
earlier build log.

**Final state:** Stages 4–6 complete and validated end-to-end. Deliverable is
a point-in-time-correct daily feature panel spanning 2010-01-04 → 2026-07-10,
plus a separate monthly Amihud file. Every stage gated by a diagnostic that
passed. Frozen research panel (no live-data refresh yet — see "Not built").

---

## What was built, in order

Each build was followed by a read-only diagnostic that had to pass before the
next stage was written. Nothing was written on top of a failing gate.

### Stage 4 — Daily PIT membership flag

Expanded the 33 semi-annual NIFTY 500 snapshots (2010-06-30 → 2026-06-01) into
a daily `(Date, ticker, in_nifty500_flag)` panel via an **explicit `as_of`
join** — the point-in-time rule (use the latest snapshot dated ≤ the current
day, never a future one) lives in one named function, not scattered through the
code.

Output: `universe/daily_pit_flag_v1_*.parquet` (3,332,879 rows).

Ratified conventions baked in:
- **Q3 tri-state flag.** `NaN` before the first snapshot (2010-06-30), never
  fabricated as `0`. Rationale: `0` asserts non-membership we cannot know, and
  backfilling would be survivorship-biased (stocks enter the index *because*
  they rose). Cost: downstream filters must use `== 1` / `== 0` explicitly,
  never `!= 1` (which bundles the 85,464 unknowns with the known-outs).
- **Q2 extrapolation tag.** The 2026-06-01 flag is held flat to the panel end
  and every row past it carries `flag_is_extrapolated=True`. The hold is sound
  (NIFTY 500 rebalances semi-annually) but the tag records that it was
  *inferred*, not *observed*.
- **Provenance carried through** via `snapshot_source`, so the 2018
  carry-forward window stays visible (99,759 rows tagged).

Diagnostic findings worth recording:
- **Flagged-vs-true count** landed at max shortfall 4, every gap named (Kwality
  Wall's, 3 Vedanta entities, Malco Energy, plus DENABANK/MIRCELECTR early).
  The pre-registered prediction was "≤5, recent snapshots only" — held. A
  shortfall in an early snapshot would have exposed a broken ISIN join; none
  appeared.
- **RANBAXY resolution limit (structural, logged not fixed).** The flag drops
  RANBAXY on 2014-12-31 while the stock traded until 2015-04-01, because
  Bloomberg's December snapshot had pre-emptively removed it (Sun Pharma
  merger). This is the general shape of the semi-annual grid: membership can be
  off by up to ~6 months in either direction. "PIT flag" here means *PIT as of
  the last semi-annual snapshot*, not as of the actual index event.
- **Two off-snapshot transitions** (AFTEK, KENNAMET) turned out to be benign
  price-gap artifacts — the flag was correct on every row; the transition was
  merely *observed* late because those tickers had no price rows on the real
  snapshot boundary. The invariant that actually matters (membership only
  changes when the governing snapshot changes) held.

### Stage 3B — Remediation of the raw price panel (v1 → v2)

A mid-pipeline correction discovered two stages downstream. The ±30% return
review list surfaced impossible overnight moves (Whirlpool +570%), which
investigation traced to **yfinance fabricating a placeholder row when a stock
does not trade** — `Open==High==Low==Close`, `Volume==0`, prior close copied
across. Three defect classes emerged:

| Mechanism | Rows | Detection |
|---|---|---|
| Phantom rows | 57,876 | `O==H==L==C AND Volume==0` (exact) |
| Unadjusted split splices | 8 | named (ticker, ex-date) pairs |
| Isolated bad prints | 6 | named (ticker, date) pairs |

Total 57,890 rows masked (OHLC → NaN; **Volume and the row retained**),
2.06% of the panel, 100% yfinance, zero Bloomberg.

Output: `raw/prices/raw_price_panel_consolidated_v2_*.parquet`. v1 retained,
never modified.

Key reasoning:
- **Fix at the raw layer, not the feature layer** — the rows are wrong at the
  source; masking downstream would leave fabricated prices for every future
  consumer to remember to handle.
- **Mask, don't delete or interpolate.** When a price is unknown, say so
  (NaN). Deleting rows would corrupt "the prior trading row"; interpolating
  would substitute a plausible lie — the exact disease. `price_status` column
  records every masked row's reason.
- **Splices: NaN the 8 boundary rows, do NOT rescale pre-history** (memory
  fact: a prior custom split-adjustment attempt double-adjusted and was
  discarded). Returns *within* each pre-split window remain correct
  (scale-invariant); only price *levels* across the boundary are incomparable
  for those 8 tickers.
- **Bad prints: 6 hardcoded rows, no detector in the pipeline.** A shape-based
  detector would trip on genuinely volatile stocks; six auditable rows beat a
  silent heuristic.

**The methodological lesson (now in the manifest):** the ±30% review list
caught 193 of ~21,472 phantom-boundary artifacts — 0.9%. Every validation up
to that point was *threshold-based*, and the contamination was *sub-threshold
by construction* (a phantom price near the resuming price produces a small
artifact). What found the full scope was *structural* — searching for a known
fabrication signature, not for large numbers. **Thresholds catch loud errors;
only structure catches quiet ones.**

### Stage 5a — Returns

`overnight = open[t]/close[t-1]-1`, `intraday = close[t]/open[t]-1`,
`daily = close[t]/close[t-1]-1`, built on the remediated v2 panel. `t-1` is the
prior *trading* row for that ticker (`groupby.shift(1)`), so holidays,
weekends, and suspensions are handled by construction. Masked rows and the rows
that follow them correctly yield NaN returns (the boundary-artifact kill).

Output: `features/returns_panel_v2_*.parquet`.

Gate: the algebraic identity `(1+overnight)(1+intraday)-1 == daily` held to
7.1e-15 across all rows — the single highest-value check, since it is
arithmetically necessary and any failure means misalignment. The ±30% list
collapsed from 193 (v1) to 151 (v2); the removed rows were phantom/splice
artifacts, the survivors real events (IMAGICAA, MVL penny ticks, JETAIRWAYS
collapse).

### Stage 5b — Turnover + realized volatility

`turnover = volume*close`; `realized_vol_60/120 = rolling std of daily_return
(full window, ddof=1) × sqrt(252)`.

Output: `features/features_daily_v2_*.parquet` (adds the vol/turnover columns).

Gate — the **warm-up assertion**: `realized_vol_120` first appears on *exactly*
trading day 121 for all clean-start tickers (927/927), never earlier for any of
the 1,069. This is the only check that distinguishes "no data yet" from
"silently backfilled," and it's why full `min_periods=window` was insisted on.
A window containing any NaN daily_return yields NaN (strict), so masked gaps
don't leak into vol.

Closing the loop on KSK: its ~30-month suspension window is now NaN vol, not
the fabricated ~0 that v1 would have produced — which would have ranked a
suspended stock as the safest name in the universe.

### Stage 5c — Monthly Amihud illiquidity

`daily_amihud = |daily_return|/turnover`; monthly mean per ticker. **Separate
monthly file** (Q6 — never broadcast onto daily rows, which would leak future
days). Zero-turnover days masked to NaN and excluded; a month needs ≥15 valid
days; `n_valid_days` retained so the threshold is reversible.

Output: `features/amihud_monthly_v2_*.parquet` (162,783 stock-months).

Gate: distribution sharply right-skewed with HDFCBANK owning the most-liquid
tail and dead micro-caps the illiquid end — the correct orientation. One
logged curiosity: VIVIDHA 2020-05 has Amihud exactly 0 (15 flat-but-traded
days), same stuck-price family as KSK/KENNAMET — deferred to anomaly review.

### Stage 6 — Join + manifest

Left-joined the PIT flag onto the daily feature panel as **metadata**
(membership never filters). Row count preserved exactly (`validate='one_to_one'`
would raise on a fan-out); every feature value bit-identical after the join.

Deliverable: `features/feature_panel_with_pit_v2_*.parquet` (3,332,879 rows,
1,069 tickers, 2010-01-04 → 2026-07-10).

`manifest.yaml` records lineage, ratified decisions, measured counts, and every
logged limitation. A read-back cell re-opens it and confirms each recorded
count matches the file — passed.

---

## Infrastructure lesson from this session (notebook robustness)

Mid-session, a stale Colab Drive mount presented as an "empty project folder,"
which triggered a long false-alarm chain (suspected wrong account, suspected
lost data). Root cause: the mount had synced the folder tree but not the file
contents. **No data was ever lost** — the files were on Drive throughout.

Two durable fixes went into the canonical notebook:
1. A **resolver** (`resolve()` / `out_path()`) that centralises every file
   path, finds the newest version by glob (date-independent), and refuses
   same-day overwrites. Reads are newest-wins; builds are dated and never
   overwrite.
2. A **preflight** that counts parquet files under the project root and fails
   loudly with a plain-language message if it sees zero ("almost always a stale
   mount — Runtime > Disconnect and delete runtime, remount"), instead of a
   cryptic error deep in a build cell.

Separately, a missing cell (the manifest *build* was never pasted into the
notebook, only the *read-back*) caused a `FileNotFoundError` that looked like a
data problem but wasn't. The canonical notebook now contains both.

---

## Not built (deliberate boundaries, for the record)

- **Data refresh / live extension.** This pipeline is a *transformation* of a
  frozen raw panel ending 2026-07-10. Running it does not fetch or append new
  dates. Bringing the panel to current is a separate stage (pull new OHLCV →
  append → re-run remediation → re-run features → refresh membership snapshot),
  with the Bloomberg human-in-the-loop wrinkle for the 114 Bloomberg tickers.
  Not needed for hypothesis development on the frozen panel.
- **Membership refresh cadence:** append the next NIFTY 500 snapshot in **early
  October** (after the September review takes effect), then early April, etc.
  Never January (that would pull stale September membership). On refresh, the
  `flag_is_extrapolated` tail flips to observed automatically.
- **Anomaly review** for the stale-price family (KSK ~30-month suspension,
  KENNAMET 861-day gap, VIVIDHA zero-Amihud) — needs a consecutive-identical-
  close detection rule, not yet built.
