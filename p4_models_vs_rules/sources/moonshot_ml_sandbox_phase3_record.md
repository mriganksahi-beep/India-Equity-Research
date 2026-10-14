# moonshot_ml_sandbox — Phase 3 record: from forecasts to portfolios

**Dates run:** 2026-10-04 (Phase 3a), 2026-10-05 (diagnostic, Phase 3b)
**Notebooks:** `moonshot_ml_sandbox_phase3a_portfolios_v2.ipynb`, `moonshot_ml_sandbox_phase3b_model_vs_twin.ipynb`
**Inputs:** V1 dataset (`ml_sandbox_v1_dataset_*.parquet`), frozen panel `feature_panel_with_pit_v2_*.parquet`, Phase 2 out-of-sample predictions (`phase2_oos_predictions_*.parquet`, saved once, never overwritten)
**Figures:** `viz_exports_phase3a_portfolios_v2/`, `viz_exports_phase3b_twins/`

---

## 1. The question

Phase 2 fitted logistic regressions that forecast four events per stock over the next 20 trading days. Phase 3 asked:

- **3a:** If each month you acted on those out-of-sample forecasts, would the portfolio be better or safer than holding everything?
- **3b:** Do the fitted models add anything beyond a hand-made sort on the traits they turned out to use?

---

## 2. What Phase 2 had found (inputs to this phase)

| Event (next 20 trading days) | Model | Out-of-sample skill* |
|---|---|---|
| Beat the median stock | M1 | 0.0027 |
| Land in the top 20% | M1 | 0.0022 |
| Crash into the bottom 20% | M1 | 0.0192 |
| Be among the 20% most volatile | M1 / M2 | 0.1292 / 0.1435 |

\*Skill = 1 − log-loss / log-loss of the intercept-only model. 0 = no better than the average rate.

**Reading:** the *size* of the next move is predictable (volatility persists, consistent with the Step 2 Markov result); its *direction* barely is. M2 (squared terms) passed the materiality rule only for high vol.

---

## 3. Method

### 3.1 How a portfolio is formed (identical every month)
1. On the first trading day of each month (date t), take every stock in the Nifty 500 that day (`in_nifty500_flag == 1`) with five features known at close t: `momentum_12_1`, `reversal_1m`, `realized_vol_60`, `log_turnover`, `overnight_drift`, each z-scored across stocks that day.
2. Each model computes `score = b + w·z` (M2 adds `v·z²`) and `P = sigmoid(score)`. Weights come from the fold trained on the previous 3 calendar years, with a 21-trading-day purge. Models are refit each January.
3. **Only the ranking is used.** Because the sigmoid is monotone, ranking by P is ranking by `w·z`; the intercept and calibration play no role.
4. With n stocks, k = round(0.20·n). "Pick" holds the k highest; "avoid" holds the n − k others.
5. Equal weights; buy at open t+1, sell at open t+21 (the window the models predicted). The month's return is the plain average of the held stocks' returns.
6. Chain the monthly returns into a wealth curve.

### 3.2 Portfolios

| Portfolio | Ranks by | Holds |
|---|---|---|
| Baseline | — | all stocks |
| Pick outperformers | P(beat median), M1 | top 20% |
| Pick big winners | P(top 20%), M1 | top 20% |
| Avoid crashes | P(crash), M1 | all but top 20% |
| Avoid wild movers | P(high vol), M2 | all but top 20% |
| Twin: cut most volatile *(3b)* | `realized_vol_60_z` | all but top 20% |
| Twin: momentum − volatility *(3b)* | `momentum_12_1_z − realized_vol_60_z` | top 20% |
| Ref: momentum only *(3b)* | `momentum_12_1_z` | top 20% |

### 3.3 Measurement
CAGR, annualised volatility, Sharpe (no risk-free rate), max drawdown, hit rate, one-way turnover. Comparisons use a **paired moving-block bootstrap**: the same months are resampled for both portfolios, in 6-month blocks, 2,000 draws.

### 3.4 Pre-committed reading rules
- **3a, avoid portfolios:** *reduce risk* if the volatility-difference CI < 0 and max drawdown is smaller than baseline; *worth it* if the Sharpe-difference CI > 0.
- **3a, pick portfolios:** *add return* if the active-return CI > 0 and active return is positive in both halves (2014–19, 2020–25).
- **3b, model vs twin (3 primary pairs):** Sharpe-difference CI (95%) entirely > 0 → model adds value; entirely < 0 → twin better; includes 0 → no detectable difference. The 98.33% CI (three pairs) is reported alongside.

---

## 4. Data-gap finding (discovered during 3a)

**What happened.** The first 3a run stopped on its own check: 142 rebalance months instead of 144. Investigation showed that **V1 has no cards from 2025-03-17 to 2025-06-17 (62 trading days)**, even though the source panel has full data for that period.

**Cause (confirmed).** The V1 builder reads `realized_vol_60` straight from the panel and drops any card with a missing feature. One missing daily return on the vendor-failure date (2025-03-18) blanks the panel's 60-day rolling volatility for the next ~60 trading days.

| Builder input | Missing before hole | Missing inside hole | Missing after |
|---|---|---|---|
| `realized_vol_60` | 0.3% | **97.4%** | 0.4% |
| `daily_return` | 0.0% | 3.2% | 0.0% |
| `overnight_return` | 0.0% | 3.2% | 0.0% |
| `turnover` / `Open` | 0.0% | 1.6% | 0.0% |

One further single-day gap (a card whose buy or sell day falls on 2025-03-18) is missing by design. No unexplained gaps.

**Handling in Phase 3.** The portfolios **sit out April, May and June 2025** (no cards, or cards starting 12 trading days late). This leaves 141 invested months. Sat-out months count as neither gain nor loss, and any other gap stops the notebook. A rebalance is never shifted deep into a month.

**Consequences.**
- All earlier phases silently skipped this quarter (about 2% of 2014–2025). Nothing was computed on wrong data; one quarter was unmeasured.
- An earlier build funnel reported 65,678 cards dropped for missing features, attributed to warm-up. About 30,000 of them were this hole. The funnel should have been broken down by date at the time.
- **Flag for `moonshot_compounding`:** `realized_vol_60` (and probably `realized_vol_120`, over a longer window) comes from the main project's feature panel, so any main-project test using it likely has the same blank.

**Fix (not yet done):** V1.1. Recompute volatility so that one missing day doesn't blank the window (e.g. require 50 of 60 days, or skip the vendor date), rebuild V1, and rerun the phases whose numbers matter.

---

## 5. Audit checks (all passed)

| Check | How | Result |
|---|---|---|
| Phase 2 reproduction | skill scores recomputed from saved predictions | match within 6e-4 |
| Leakage, construction | each prediction lies in its fold's test year; last training card's sell day < test start | pass, all folds |
| Leakage, refit | crash M1 for 2019 refit from scratch on pre-2019 rows only | max diff 0.0 vs stored |
| Universe | every candidate has flag = 1 on its rebalance date | pass; e.g. 63MOONS.NS excluded after leaving 2017-01-02 |
| Forward return | 3 returns recomputed from raw open prices | exact match |
| Portfolio returns | 2 months × 5 portfolios recomputed with `nlargest` | exact match |
| Twins | 2 months recomputed by `nlargest`/`nsmallest` | exact match |
| Hidden disasters | in-index stocks whose data ends within 21 days | 23 stock-months = 0.033%; median +3.0% to last price → small |
| Coverage | usable cards on rebalance dates | 100%; median 490 stocks/month (min 410) |
| Holding windows | adjacent-month overlap / uncovered days | overlap mean 0.29 (max 3); uncovered mean 0.88 (max 3) |

---

## 6. Results — what the code found

### 6.1 Phase 3a: model portfolios vs holding everything (141 months, 2014–2025)

| Portfolio | CAGR | Vol | Sharpe | Max DD | Turnover | CAGR 2014–19 | CAGR 2020–25 |
|---|---|---|---|---|---|---|---|
| Baseline | 12.6% | 21.3% | 0.67 | −45.4% | 0.017 | 8.3% | 17.3% |
| Pick outperformers | 17.9% | 16.3% | 1.10 | −26.4% | 0.350 | 16.0% | 19.9% |
| Pick big winners | 11.4% | 23.3% | 0.59 | −52.4% | 0.334 | 5.3% | 18.1% |
| Avoid crashes | 15.2% | 19.8% | 0.82 | −36.1% | 0.085 | 12.3% | 18.3% |
| Avoid wild movers | 14.9% | 19.7% | 0.81 | −37.5% | 0.080 | 11.8% | 18.1% |

| vs baseline | Active return (95% CI) | Halves | Vol diff CI | Sharpe diff CI | Verdict |
|---|---|---|---|---|---|
| Pick outperformers | +3.72% [−1.60, +8.67] | +6.0% / +1.3% | [−6.9, −2.7]% | [+0.21, +0.73] | adds return: **no** |
| Pick big winners | −0.60% [−4.69, +3.08] | −2.4% / +1.3% | [+0.6, +3.8]% | [−0.25, +0.07] | adds return: **no** |
| Avoid crashes | +1.99% [+0.40, +3.66] | +3.4% / +0.6% | [−1.9, −1.2]% | [+0.08, +0.26] | reduces risk: **yes**; worth it: **yes** |
| Avoid wild movers | +1.69% [+0.18, +3.21] | +2.9% / +0.4% | [−2.0, −1.2]% | [+0.07, +0.24] | reduces risk: **yes**; worth it: **yes** |

**Stricter CIs (98.75%, four portfolios):**
- Sharpe gains survive for both avoid portfolios and for Pick outperformers.
- Extra return survives only for Avoid crashes, and only just (lower bound +0.05%).

**Held vs set aside (monthly averages):**

| | Crash rate held / other | Return held / other |
|---|---|---|
| Pick outperformers | 16.0% / 20.9% | 1.50% / 1.11% |
| Pick big winners | 25.4% / 18.6% | 1.14% / 1.20% |
| Avoid crashes | 17.5% / 29.8% | 1.35% / 0.52% |
| Avoid wild movers | 17.5% / 29.8% | 1.33% / 0.63% |

**Identity check.** Because the baseline holds everything:
- avoid: held − baseline = 0.2 × (held − cut)
- pick: held − baseline = 0.8 × (held − rest)

These reproduce the reported active returns (Avoid crashes ≈ 2.0%/yr, Pick outperformers ≈ 3.7%/yr, Pick big winners ≈ −0.6%/yr).

### 6.2 Diagnostic: what the rankings actually sort on

| Average z-score of held stocks | Momentum | Reversal | Volatility | Turnover | Overnight drift |
|---|---|---|---|---|---|
| Pick outperformers | +0.83 | −0.01 | −0.64 | +0.19 | −0.39 |
| Pick big winners | +0.96 | +0.07 | +1.07 | +0.11 | +0.13 |
| Avoid crashes | +0.06 | −0.04 | −0.36 | −0.02 | −0.11 |
| Avoid wild movers | 0.00 | −0.02 | −0.37 | +0.01 | −0.10 |

- The avoid portfolios hold 80% of stocks, so the cut group averages about −4 × the held value: **about +1.45 on volatility**.
- Stocks cut by Avoid crashes that Avoid wild movers also cuts: mean **86.2%** (min 73.7%).
- Rank correlations between predictions: P(crash)–P(high vol) **0.93**; P(outperform)–P(crash) **−0.84**; P(outperform)–P(high vol) −0.72; P(top)–P(high vol) 0.60; P(outperform)–P(top) −0.04.

### 6.3 Phase 3b: model vs no-model twin

| Portfolio | CAGR | Vol | Sharpe | Max DD | Turnover | Cost/yr at 0.2%/side (assumed) |
|---|---|---|---|---|---|---|
| Twin: cut most volatile | 14.8% | 20.0% | 0.80 | −37.0% | 0.080 | 0.4% |
| Twin: momentum − volatility | 21.3% | 17.2% | 1.22 | −25.2% | 0.286 | 1.4% |
| Ref: momentum only | 19.6% | 20.7% | 0.98 | −29.8% | 0.226 | 1.1% |

| Model vs twin | Holdings overlap | Sharpe diff | 95% CI | 98.33% CI | Verdict |
|---|---|---|---|---|---|
| Avoid crashes vs cut most volatile | 95.8% | +0.025 | [+0.003, +0.060] | [−0.001, +0.068] | **model adds value** (marginal) |
| Avoid wild movers vs cut most volatile | 96.9% | +0.011 | [−0.008, +0.041] | [−0.013, +0.048] | **no detectable difference** |
| Pick outperformers vs momentum − volatility | 74.3% | −0.119 | [−0.244, −0.040] | [−0.283, −0.019] | **twin is better** |
| *(ref)* Pick outperformers vs momentum only | 48.2% | +0.122 | [−0.076, +0.336] | [−0.110, +0.386] | — |

Relative wealth (model ÷ twin) over the period:
- Avoid crashes ends near **1.045×**, rising slowly.
- Avoid wild movers ends near **1.01×**.
- Pick outperformers falls **steadily to about 0.72×**. The decline is persistent, not a single episode.

### 6.4 Expectations (written before each run) vs outcome

| Expectation | Outcome |
|---|---|
| Avoid portfolios: clearly lower vol and drawdown, similar or slightly better return | ✓ |
| Pick outperformers: small gain at best, mostly before 2020 | timing ✓, size larger than expected |
| Pick big winners: more volatile, worst risk-adjusted | ✓ |
| Avoid portfolios vs twin: high overlap, no difference | ✓ (crash model a sliver ahead) |
| Pick outperformers vs twin: probably no difference | ✗ — twin better |

---

## 7. Interpretation — what I think it means

*(Separate from the outputs above; these are readings, not results.)*

1. **The avoid models are essentially "cut volatile stocks".** Their rankings nearly coincide with each other (0.93) and with trailing volatility (96–97% overlap). The high-vol model rediscovered volatility persistence. The crash model adds at most about 0.4%/yr, which does not survive the stricter bar.
2. **The four models live on a two-axis map: momentum and volatility.**
   - Pick outperformers = trending and calm.
   - Pick big winners = trending and wild. The volatility that feeds the good tail also feeds the bad one, which is why its picks crashed more often.
   - The avoid rules cut the wild end.
3. **Weak forecasts can still make useful portfolios.** The outperform model barely predicted its own question (skill 0.0027). But its ranking sorted stocks into a calm, trending corner that roughly kept the market's return with much less risk. Most of its Sharpe gain came from lower volatility, not higher return.
4. **The model found the right ingredients but wrote a worse recipe than 1:1.** Plausible, untested reasons:
   - **Objective mismatch.** It was trained on log-loss for P(beat median) across all stocks, but the portfolio only uses the top-20% ranking.
   - **Estimation noise.** Annual refits on 3 years let weights wobble, and turnover is higher (0.35 vs 0.29).
   - **Noisy extra traits.** Weight spread onto overnight drift and turnover.
   - This fits the general pattern that simple equal-weight combinations often beat estimated weights out of sample.
5. **Overall:** the logistic models were more valuable as a microscope than as a strategy. They identified which traits matter, and the simplest use of those traits performed as well or better.
6. **2020–21 is consistent with momentum risk.** Pick outperformers lost 8.6% and 5.4% vs baseline in those years, a sharp rebound followed by a rally led by beaten-down stocks. That is the kind of turning point where momentum tilts typically suffer. Not tested.

---

## 8. Caveats and things not to over-read

- **No trading costs.** At an assumed 0.2% per side: Pick outperformers ≈ 1.7%/yr (about half its active return); momentum − volatility twin ≈ 1.4%/yr; avoid rules ≈ 0.4%/yr.
- **Equal weighting across the Nifty 500** gives small, less liquid stocks the same weight as large caps. Capacity and impact are not modelled.
- **The momentum − volatility twin was designed with hindsight** (after seeing the model's full-period tilts). Its 21% CAGR and Sharpe 1.22 are "promising, unverified".
- **No untouched holdout.** 2014–2025 has now been used for many tests in this sandbox. Every number here is development-period evidence. A true out-of-sample check needs data not yet used, e.g. forward paper tracking from now.
- **Multiple testing.** Seven portfolios and several comparisons. Only results that survive the stricter CIs should be treated as established.
- **Edges shrink after 2020** for the avoid rules (active return 3.4% → 0.6%/yr).
- **2025 has 9 invested months** (data gap).

---

## 9. Open issues

1. **V1.1 rebuild** with NaN-tolerant volatility; then rerun the phases whose numbers matter.
2. **Flag to `moonshot_compounding`:** check every test using `realized_vol_60` / `realized_vol_120` for the 2025 vendor-date blank.
3. **Stress-test the momentum − volatility twin** before believing it: cost sensitivity, size and liquidity buckets, stability across nearby weights and cut sizes. Judge on stability, never pick the best.
4. **Test the objective-mismatch explanation:** a model trained to rank returns, judged against the twin.
5. **Holdout policy:** decide what counts as the final out-of-sample check (forward paper tracking).

---

## 10. Log entries

```
id: sandbox-phase3a-logit-portfolios
date: 2026-10-04
hypothesis: Ranking stocks monthly by Phase 2 out-of-sample logistic probabilities (20% cut) gives better or safer portfolios than equal-weight-all
category: other (model-driven selection)
mechanism_tags: volatility_persistence, low_volatility, momentum, crash_avoidance
data_and_universe: V1 cards 2014-01..2025-12, 141 monthly rebalances (2025-04..06 sat out, V1 gap); PIT Nifty 500 (flag==1 on rebalance date); open t+1 -> open t+21
result: Sharpe baseline 0.67; Avoid crashes 0.82 (diff CI [+0.08,+0.26]); Avoid wild movers 0.81 ([+0.07,+0.24]); Pick outperformers 1.10 ([+0.21,+0.73], active CI [-1.6%,+8.7%]); Pick big winners 0.59
validation_summary: coverage pass (100%, median 490 stocks); spot-check pass (returns from raw opens, nlargest recompute); leakage pass (fold construction + refit 0.0 diff); universe pass (63MOONS.NS example)
holdout_status: not yet tested
decision: needs modification (benchmark vs simple factor sorts -> Phase 3b)
insight: Weak directional forecasts still improved risk-adjusted returns, because their rankings sorted stocks along volatility and momentum.
incidental_observations: V1 has no cards 2025-03-17..06-17 because panel realized_vol_60 is blank for ~60 days after the 2025-03-18 vendor failure; crash and high-vol rankings correlate 0.93; avoid-rule edges shrink after 2020; Pick outperformers lost in 2020-21.
flag_for_idea_generator: Is the post-2020 shrinkage of the low-volatility edge structural (market composition) or regime-specific? Does momentum's 2020-21 drawdown align with identifiable regime transitions?
related_ids: sandbox-step2-vol-markov, sandbox-phase2-logistic
```

```
id: sandbox-phase3b-model-vs-twin
date: 2026-10-05
hypothesis: The Phase 3a model portfolios add value beyond hand-made sorts on the traits they load on (volatility; momentum - volatility)
category: other (model vs simple benchmark)
mechanism_tags: low_volatility, momentum, estimation_noise, objective_mismatch
data_and_universe: same as sandbox-phase3a-logit-portfolios
result: Sharpe diff model-twin: Avoid crashes +0.025 [+0.003,+0.060]; Avoid wild movers +0.011 [-0.008,+0.041]; Pick outperformers -0.119 [-0.244,-0.040] (twin Sharpe 1.22, CAGR 21.3%)
validation_summary: coverage pass; spot-check pass (twins recomputed via nlargest/nsmallest); leakage pass (twins use only date-t z-scores; Phase 3a reproduced exactly); universe pass (inherited)
holdout_status: not yet tested
decision: needs more data (twin stress test; ranking-objective model)
insight: The logistic models identified the right factors, but a fixed 1:1 combination used them better than the fitted weights did.
incidental_observations: twin momentum-volatility was designed with hindsight; its strength (Sharpe 1.22) needs cost/size/stability checks; momentum-only Sharpe 0.98 sits between model and twin.
flag_for_idea_generator: Does training on the portfolio's actual objective (ranking top-quintile returns) close the gap to the fixed-weight twin, or are fixed weights simply more robust here?
related_ids: sandbox-phase3a-logit-portfolios, sandbox-phase2-logistic
```
