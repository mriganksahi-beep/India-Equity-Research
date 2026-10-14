# Momentum — Test 1 findings

id: momentum_12_1_decile_ls
date: 2026-07-27
hypothesis: Cross-sectional 12-1 momentum — high trailing-return stocks outperform low over the next month (harness-validation test)
category: momentum
mechanism_tags: cross_sectional, price_momentum, decile_long_short, equal_weighted

data_and_universe: >
  Nifty 500 PIT universe, v3 panel (feature_panel_with_pit_v2 derived from
  raw_price_panel_consolidated_v3; 3,328,132 rows, 2010-01-04 to 2026-07-10).
  Signal = compounded daily_return over [t-252, t-21] (12-1 momentum). Monthly
  rebalance on the last trading day. PIT filter in_nifty500_flag == 1. Strict
  232/232 clean window, vendor-failure dates (2025-03-18) excluded from the
  requirement. Months with <100 eligible names skipped.
  IMPORTANT: the eligible universe is "Nifty 500 names with a clean trailing
  year", NOT the full Nifty 500 — in distress-heavy years the 232 filter
  removes many flagged-but-untradeable names (see 2024 finding in memory doc).

result: >
  Long-short D10-D1, equal-weighted, GROSS of costs. 179 monthly obs.
  Mean spread 1.740%/mo (95% block-boot CI
  [0.769%, 2.705%]).
  Annualised Sharpe 0.847 (95% CI
  [0.335, 1.503], P(Sharpe>0)=1.000).
  Hit rate 64.8%. Decile monotonic steps 7/9.
  Mean IC 0.0605 (95% CI [0.0350, 0.0865]),
  IC info ratio 0.347.
  Liquidity terciles: T1illiq=1.97%/mo; T2mid=2.13%/mo; T3liq=1.03%/mo.
  2020 crash-window max drawdown: -46.7%.

validation_summary: >
  coverage: eligible universe ~430-490/mo normal years, 19 months skipped (<100).
  spot-check: leakage self-test passed (window strictly before rebalance, gap>=21d).
  leakage: 0 forward-looking violations by construction (forward window starts t+1).
  universe: PIT filter == 1 applied; RANBAXY/HDFC transitions correct (Phase A).

holdout_status: n/a (dev-period baseline; no holdout split defined for this harness test)
decision: [FILL IN after reading — expected: promoted/reject/needs modification]
insight: [FILL IN — one sentence, independent of pass/fail]

incidental_observations: >
  [REQUIRED — fill in from the run. Candidates to check: is the spread
  monotonic or driven by extremes? Does it concentrate in T1 (illiquid)? Any
  decade where it decays? Anything odd in the IC series? 2018 H2 outlier?]

flag_for_idea_generator: >
  [If any incidental seeds a new hypothesis, phrase it here as a gap/question.]

related_ids: [reversal (shares daily_return + PIT flag; opposite sign at short horizon)]
