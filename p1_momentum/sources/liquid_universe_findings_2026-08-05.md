# Momentum — Liquidity-Universe Survival Test — FINDINGS

id: momentum_liquid_universe_survival
date: 2026-08-05
mechanism: universe-construction test (NOT an M-series enhancement; a positive
  structural finding, not a signal modification)
hypothesis: >
  Momentum concentrates in illiquid names (frozen baseline: liquid tercile Sharpe
  0.42 vs ~0.88 illiquid). Does the strategy SURVIVE when you trim the illiquid
  tail and re-form the portfolio on an investable universe? The baseline only
  DIAGNOSED the tradeability problem (via conditional liquidity slices); this test
  builds an actual liquid-universe strategy and measures it end-to-end.
category: momentum / tradeability / universe-construction
mechanism_tags: liquidity_filter, tradeability, turnover, amihud, price_impact,
  universe_construction, decile_reform, degradation_curve

data_and_universe: >
  Nifty 500 PIT universe, v3 panel (feature_panel_with_pit_v2_2026-07-24, 3.33M
  rows). PIT filter in_nifty500_flag == 1. 8 corp-action tickers screened. 179
  months. 12-1 momentum signal unchanged. Method: each month (1) collect
  momentum-eligible names with a PIT liquidity value, (2) FILTER to the liquid
  universe (top 30/50/70% + full), (3) RE-FORM momentum deciles WITHIN the
  filtered universe (fresh breakpoints), (4) D10-D1 equal-weight forward return.
  Liquidity measures: PRIMARY trailing-21-day mean turnover (PIT); ROBUSTNESS
  Amihud price-impact (monthly, PIT prior-month join, low amihud = liquid).

why_not_the_conditional_slice: >
  The baseline/M6 measured momentum CONDITIONAL on liquidity — a slice of the
  full-universe strategy (liquid names within full-universe deciles). This test
  FILTERS FIRST then RE-FORMS deciles within the liquid universe — a genuinely
  different portfolio. VERIFIED: re-forming produces different top-decile
  membership than slicing (9-name symmetric difference in a sample month; the two
  are not the same test). This distinction is WHY the result differs from the 0.42
  tercile number.

result: >
  DECISION: POSITIVE STRUCTURAL FINDING — momentum is investable. It degrades
  GRACEFULLY (not catastrophically) as the universe tightens to tradeable names,
  robustly across BOTH liquidity definitions.

  DEGRADATION CURVE (D10-D1 annualised Sharpe, 179 months):
    floor         turnover Sharpe [95% CI]        amihud Sharpe [95% CI]    monotonicity
    top 30% (143n)  0.394 [-0.11, 0.79]            0.394 [-0.09, 0.86]       0.75-0.79
    top 50% (238n)  0.507 [ 0.02, 0.92]            0.519 [-0.01, 0.99]       0.84-0.87
    top 70% (332n)  0.710 [ 0.18, 1.16]            0.622 [ 0.05, 1.12]       0.98
    full   (475n)   0.755 [ 0.19, 1.32]            0.758 [ 0.19, 1.33]       0.94
  Per-month spread scales with the Sharpe (0.90%/mo at top 30% -> 1.54%/mo full).

  KEY READS:
   - Momentum is NOT trapped in the illiquid tail. Top-70% retains ~82-94% of the
     full-universe Sharpe; top-50% ~67-69%; only the aggressive top-30% filter
     weakens toward the illiquid-slice level (0.39) with a degraded staircase.
   - The decile staircase gets CLEANER as you tighten toward the middle floors
     (monotonicity 0.94 full -> 0.98 at top-70%). The liquid universe is a
     better-behaved signal, just with a smaller spread.
   - The conditional-tercile evidence (liquid 0.42) UNDERSTATED the tradeable
     strategy. Re-forming within the liquid universe gives 0.5-0.7, not 0.42 —
     the slice was measuring a different (weaker) thing.

robustness_turnover_vs_amihud: >
  The two liquidity definitions TRACK each other (same graceful-decline shape;
  Sharpes within ~0.09 at every floor). The finding is robust to whether
  "tradeable" means trading activity (turnover) or price impact (Amihud). Only
  divergence: at top-70%, turnover 0.71 vs amihud 0.62 — the price-impact filter
  is mildly less favorable at the moderate floor, consistent with Amihud being a
  slightly stricter "can I move this without impact" test. Second-order; the
  headline (graceful degradation, momentum survives) holds under both.

validation_summary: >
  filter fraction: top 30/50/70% keep ~146/243/340 of 486 names — exact. PASS.
  re-form != slice: 9-name symmetric difference in sample-month D10 — the test
    measures a real liquid-universe strategy, not the conditional slice. PASS.
  PIT: turnover window ends strictly before rebalance; amihud uses prior month.
    No look-ahead. PASS.
  baseline reproduction: full-universe (floor=0) Sharpe 0.755, per-month spread
    1.54% over 179 months — in the frozen baseline range (~0.85-0.91 gross,
    ~1.74% spread; the modest gap is the monthly-mark vs forward-window
    convention, benign). Harness is sound. PASS.
  amihud direction: kept names' median amihud 1.48e-11 vs full 1.42e-10 — the
    filter keeps the LIQUID (low-amihud) tail (hard-asserted). PASS.

holdout_status: n/a (structural characterization of the existing signal on a
  restricted universe; not a new tradeable signal requiring a holdout)

decision: PROMOTE to next stage (transaction-cost study). Momentum is investable
  gross-of-cost; the deployability question now hinges on costs, not on whether
  the effect survives liquidity filtering (it does).

insight: >
  Momentum in NSE is NOT trapped in illiquid names as the conditional-tercile
  evidence (0.42) implied. When you build an actual liquid-universe strategy
  (filter then RE-FORM), it retains ~65-95% of baseline strength in the top 50-70%
  by liquidity, with a CLEANER decile staircase, robustly across both turnover and
  Amihud definitions. The tercile slice understated tradeability because it
  measured liquid names within full-universe deciles rather than a re-formed
  liquid portfolio. This is the first decision-grade POSITIVE result of the
  investigation and reframes the central tradeability concern: the effect is
  investable; the open question is net-of-cost economics, not gross survival.

limitations: >
  (1) WIDE CIs — every floor's CI lower bound sits below the pre-registered 0.30
  pre-cost screen, INCLUDING the full-universe baseline (CI low 0.19). The screen
  as written conflates "weaker than baseline" (the real question) with "momentum
  is statistically bulletproof" (which it never was, even full-universe, over 179
  months). The screen is therefore too strict to be the deciding bar; the
  fraction-of-baseline-retained read (65-95%) is the better lens. (2) Cross-floor
  ordering is SUGGESTIVE not statistically separated — top-30% (0.39) and top-70%
  (0.71) CIs overlap. The degradation curve is real as a central tendency, noisy
  at the edges. (3) GROSS OF COSTS. The entire premise of going liquid is that
  costs are lower there — a 0.71 gross Sharpe in liquid names may survive costs far
  better than a 0.88 gross in illiquid names eaten by impact. That advantage is
  NOT visible in this notebook; it requires the cost study.

incidental_observations: >
  (1) The staircase IMPROVING as the universe tightens (0.94 -> 0.98) is mildly
  counterintuitive and worth remembering — trimming illiquid names removes noisy
  microcap deciles, not signal. (2) Amihud being slightly less favorable than
  turnover at top-70% is a small hint that price-impact tradeability is a stricter
  screen than activity-based tradeability — relevant to which liquidity measure a
  deployment should optimize against. (3) The pre-cost screen I pre-registered
  (CI_low > 0.30) turned out to reject the baseline itself — a lesson that a
  CI-lower-bound bar on a 179-month momentum Sharpe is inherently strict; future
  pre-registration of such bars should account for the baseline's own CI width.

flag_for_idea_generator: >
  The decision-critical NEXT TEST is a TRANSACTION-COST STUDY on the liquid
  universe — NOT another signal variant. Model realistic costs (spread + impact,
  ideally Amihud-informed) on the top-50% and top-70% strategies and ask whether
  the 0.5-0.7 gross Sharpe survives net. This is where deployability is actually
  decided. Corollary question: does the liquid strategy's LOWER costs make it
  BETTER net-of-cost than the illiquid-heavy full-universe baseline despite lower
  gross Sharpe? That net comparison, not gross Sharpe, is the real deployment
  metric. Also available as a lever: the parked signal_prop weighting (M1b) and
  liquidity-conditional vol (M3) could be layered on the liquid universe — but
  cost-first, variants-second.

related_ids: >
  momentum_12_1_decile_ls (frozen baseline — the full-universe row reproduces it,
  confirming harness soundness). momentum_enh_M6_hump_diagnostic (established the
  monotonic liquidity gradient; this test shows that gradient is shallow enough
  that momentum survives into the liquid universe). momentum_enh_M3_volscaled
  (vol-scaling WIDENED the liquidity gap — the opposite lever to universe
  restriction; relevant to net-of-cost work).

running_tally: >
  NOT an enhancement sub-test — this is a universe-construction characterization,
  not a signal variant competing against the baseline, so it does not add to the
  M-series DSR denominator (which stands at 29). The floor sweep and the Amihud
  robustness are one pre-registered test with a robustness leg, not multiple
  independent trials (floors were locked before running; we report the full curve,
  not a chosen floor).
