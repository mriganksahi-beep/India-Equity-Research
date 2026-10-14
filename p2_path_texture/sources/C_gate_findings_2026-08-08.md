# Test C — Low-Volatility Anomaly — C-GATE — FINDINGS (SHELVED)

id: microstructure_C_lowvol_gate
date: 2026-08-08
mechanism: Test C gate (Microstructure & Anomaly batch) — gated, did not reach build
hypothesis: >
  A diversified, large-cap-tilted low-vol sort improves the combined book's
  net-of-cost liquid Sharpe. The GATE decides whether low-vol is tradeable in the
  Indian liquid core or collapses to the illiquid corner (which would also
  mechanically explain the M3 scar). Gate = corr(RV, Amihud) full universe vs
  liquid core; RV = realized_vol_60.
category: low-volatility / microstructure / gate
mechanism_tags: low_vol_anomaly, realized_vol, amihud, liquidity_entanglement,
  size_control, partial_correlation, M3_scar, gated_testing

data_and_universe: >
  Nifty 500 PIT panel, v3 (feature_panel_with_pit_v2_2026-07-24). TRAINING 2010-2021
  ONLY (hold-out 2022-2026 walled off in code; vol-corruption window lies inside the
  hold-out, so training sees only clean realized_vol_60). 133 training months. 8
  corp-action tickers screened. Liquid core = top 50% by trailing-21d turnover
  (matches the liquidity-universe notebook's tradeable core). Spearman cross-
  sectional correlations per month, averaged.

result: >
  DECISION: SHELVE C (gate came back ambiguous; two pre-registered diagnostics
  resolved it decisively toward "genuine, pervasive, size-independent RV-liquidity
  entanglement"). No C-build.

  GATE (per-month Spearman, averaged over 133 training months):
    corr(RV, Amihud) FULL universe:  +0.088  (near zero)
    corr(RV, Amihud) LIQUID core:    +0.303  (ambiguous band 0.20-0.40)
    corr(RV, turnover) FULL:         +0.046
    corr(RV, turnover) LIQUID:       -0.081
  Gate rule (pre-stated): <=0.20 proceed, >=0.40 shelve, between -> surface. The
  +0.303 landed in the ambiguous band -> ran two pre-registered diagnostics.

  DIAGNOSTIC 1 (shape): mean Amihud-rank across liquid-core RV deciles rose PERFECTLY
  MONOTONICALLY (0.36 0.39 0.43 0.46 0.49 0.52 0.54 0.56 0.62 0.65; Spearman +1.000,
  no corner spike). -> the +0.303 is a REAL pervasive link, not a corner artifact.
  DIAGNOSTIC 2 (size-control): partial corr(RV, Amihud | price) = +0.242 vs raw
  +0.303 -> only 20% collapse under size control. -> the link is GENUINE
  RV-liquidity, NOT a size effect the C size-double-sort would neutralize (only a
  fifth is size).
  COMBINED: both point to real/pervasive/size-independent -> low-vol is genuinely
  liquidity-entangled even among liquid names -> SHELVE.

green_field_finding: >
  THE SURPRISING REVERSAL (the real green-field number). corr(RV, Amihud) is
  NEAR-ZERO in the full universe (+0.088) but POSITIVE in the liquid core (+0.303) —
  the OPPOSITE of an illiquid-corner effect. An illiquid-corner mechanism would give
  a strong full-universe correlation (driven by the tail) that WEAKENS in the liquid
  core; we see the reverse. So India's low-vol/liquidity entanglement lives in the
  LIQUID names, not the illiquid tail: among liquid stocks, low-vol names are the
  MOST liquid (lowest Amihud), a clean monotone gradient; the relationship washes
  out across the messy full universe. This CONTRADICTS the intuitive M3
  illiquid-corner mechanism — M3's tradeability damage did NOT come from tilting
  into an illiquid corner (that corner shows no RV-Amihud link).

  MECHANISM CAVEAT (tentative, not proven): WHY the full-universe correlation washes
  out is most plausibly that the illiquid tail's erratic/stale Amihud values dilute
  it, leaving the genuine liquid-core relationship visible only when the tail is
  removed. Stated as a hypothesis, not established — the gate tested one correlation
  pair and its diagnostics, not the dilution mechanism directly.

validation_summary: >
  hold-out wall: gate computed on 2010-2021 only; 2022-2026 untouched. PASS.
  RV: realized_vol_60 (not 120 — corruption); corruption window inside hold-out,
    training clean. PASS.
  estimator: Spearman (rank-based, robust to Amihud's heavy tail). Diagnostics
    verified on known-answer synthetic scenarios before use (real-link scenario
    read GENUINE/0%-collapse; price-mediated scenario showed the partial-corr
    collapse.) PASS.

holdout_status: n/a (shelved on training-side gate; hold-out not consulted)

decision: SHELVE (genuine size-independent RV-liquidity entanglement in the liquid
  core; a low-vol sort would carry the liquidity contamination the batch is built
  to avoid; the load-bearing output Amihud-loading diagnostic is predicted to fail)

insight: >
  Low-vol is NOT cleanly tradeable in the Indian liquid core: realized-vol and
  price-impact (Amihud) are genuinely, monotonically, size-independently entangled
  among liquid names. A diversified size-controlled low-vol sort would still tilt
  toward the lowest-impact names — the exact liquidity contamination the batch's
  vol-/liquidity-orthogonal thesis forbids. The gate + 1 diagnostic (2 trials)
  reached this for a fraction of a full build's cost, and produced a genuinely
  novel structural fact: India's low-vol/liquidity link lives in the LIQUID core,
  not the illiquid tail, reframing the M3 scar away from the illiquid-corner story.

incidental_observations: >
  (1) The gate's auto-printed M3 reading ("not purely an illiquid-corner artifact")
  UNDERSTATED the finding — the full-universe correlation isn't just "similar," it's
  near-zero, and the liquid-core one is HIGHER, an active reversal. Read the raw
  numbers, not just the gate's canned verdict. (2) D2's binary "50% collapse" cutoff
  was softened to a graded three-band reading during the build because the exact
  collapse fraction (here 20%) is the informative object, not a pass/fail. (3) The
  monotone D1 gradient (Spearman +1.000) is unusually clean — worth noting that the
  RV-Amihud link in the liquid core is one of the tightest cross-sectional
  relationships seen in the whole investigation.

flag_for_idea_generator: >
  The liquid-core RV-Amihud reversal is a green-field structural fact worth its own
  line: in India, LOW realized-vol liquid stocks are the MOST liquid (lowest price
  impact), monotonically and size-independently. Any future low-risk / low-vol idea
  must confront that entanglement head-on — it is not scrubara-ble by a size sort.
  Also reframes the M3 post-mortem: the mechanism of M3's tradeability damage is NOT
  the illiquid-corner tilt we assumed; the actual channel (how vol-scaling interacts
  with the liquid-core RV-Amihud gradient) is an open question if anyone wants to
  reopen M3.

related_ids: >
  momentum_enh_M3_volscaled (the scar this gate was meant to explain — and partly
  RE-frames: the illiquid-corner mechanism is contradicted). momentum_liquid_universe_
  survival (same liquid-core definition, top-50% turnover). Test A (young-collapse
  down-weighting was the intended C complement — now moot).

running_tally: >
  Tally 31 (C-gate = trial 30; the D1/D2 diagnostic bundle interrogating the
  ambiguous gate = trial 31, one increment, same convention as M3/M5 probes). C
  shelved without a build, so no further C sub-tests. All feed the shared-hold-out
  DSR denominator at reveal.
