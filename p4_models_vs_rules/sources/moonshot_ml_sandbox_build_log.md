# moonshot_ml_sandbox — Build Log

**Purpose of this document:** a fast-lookup reference for anyone (human or AI)
picking up this learning-project without re-reading the full session history. It
records the *concepts learned*, the *math*, the *code artifacts*, and the
*empirical findings*, woven together, plus the roadmap ahead. Treat it as a
living "what we did, why, and what we found" briefing. Companion to the
`moonshot_compounding` project (this is the ML learning sandbox sibling).

---

## Project purpose, one paragraph

A self-contained learning sandbox that uses the frozen NSE Nifty 500 feature
panel as a dataset to learn vector math → linear regression → probability →
model building, hands-on, on data whose pathologies are already understood. It is
**a learning project, not a research/strategy project**: no DSR budget, no
hold-out spent, not logged as hypotheses. The one discipline retained in full is
**leakage control** — because a leak makes the *learning* fake, not just a signal
fake. Everything reads the frozen `moonshot_compounding` v3 panel READ-ONLY and
writes only into `moonshot_ml_sandbox/`.

**Location:** `/content/drive/MyDrive/moonshot_ml_sandbox/` (Colab + Drive),
sibling to `moonshot_compounding/`.

---

## The dataset (V1) — the foundation everything sits on

**One card = one (stock, date).** Built from the frozen v3 panel. Each card:
- **5 features**, cross-sectionally z-scored within each date, all computed
  through **close(t)**: `momentum_12_1` (cumulative daily return t-252→t-21,
  skipping last month), `reversal_1m` (last 21 days), `realized_vol_60` (panel
  column), `log_turnover` (log of turnover), `overnight_drift` (trailing 21d mean
  overnight return).
- **1 target**: `fwd_ret_20d` = absolute forward return, **open(t+1) → open(t+21)**.
- **The leakage wall:** features ≤ close(t); target starts open(t+1); one-day gap
  preserved. This is *the* load-bearing invariant.

**Build facts (do not relearn):**
1. Momentum/reversal compound daily returns as `prod(1+r)-1`; missing days
   (incl. the 2025-03-18 vendor-failure date) are **dropped from the product,
   never zero-filled** — a zero is a fabricated claim the stock didn't move.
2. Vendor-failure date (2025-03-18) is excluded from any window and refused as a
   target endpoint.
3. Universe filter is `in_nifty500_flag == 1` (explicit `== 1`, tri-state safe —
   never `!= 0`, which would bundle pre-2010 NaN "unknowns").
4. Dates with `< 30` eligible names dropped (z-score across a handful is noise).
5. Cross-sectional z-scoring done **after** all filtering, so moments reference
   the surviving set.

**Final V1 dataset:** ~1.81M cards, 1,041 tickers, 2011-01-03 → 2026-06-10.
Columns: ticker, date, 5 raw features, 5 z-scored features (`*_z`), `fwd_ret_20d`,
and thinness bookkeeping (`mom_n_used`, etc.). Validated: leakage wall proven
(feature date < target date), spot-checks recomputed from raw prices to `0.00e+00`.

**Schema gotcha:** the source v3 panel columns are Capitalized (`Date`, `Open`,
`Close`, `Volume`) and carry a `source` column (yfinance/bloomberg). Open coverage
97.95% (Bloomberg 94.1%, yfinance 98.3%) — enough to keep the open→open target.

**Code artifact:** `moonshot_ml_sandbox_v1_builder.ipynb` (mount → load →
filter → per-ticker feature/target engine → NaN & thin-date drops → z-score →
6 validation blocks → write). The per-ticker engine is an explicit loop (not
vectorized) for verifiability of the window/vendor bookkeeping.

---

## Concepts learned (the toolkit), grouped

### Data integrity
- **Look-ahead bias & the leakage wall.** Features through close(t), outcome from
  open(t+1); measured at separated times, always. Leakage is silent — build the
  wall regardless.
- **Point-in-time correctness.** Any quantity used for a decision must have been
  knowable then. Drove trailing-252d decile bins and the purge gap.
- **NaN = unknown.** Never fill with a plausible number; a non-existent date is
  deleted, not zeroed. Conflating "unknown" and "didn't happen" corrupts silently.

### Distributional foundations
- **Random variable / distribution / moments.** A future return is a random
  variable; its distribution is summarized by 4 moments — mean (center),
  variance=volatility (spread), skew (lopsidedness), kurtosis (tail-fatness).
  **Regression predicts only the 1st moment.** Fork 3 = deciding to predict the
  higher moments.
- **Fat tails / non-Normality.** Returns are NOT bell-curved (momentum kurtosis
  ≈ 35). Extreme moves far more common than Normal predicts. Tails = where risk
  and ruin live.
- **Log transform.** Heavily right-skewed positive quantities (turnover, range
  61 → 3.8e11) are logged to compress into a usable near-symmetric shape.
- **LLN & CLT.** Averages converge to truth; sums tend to Normal — BUT both
  assume independence + finite variance, which finance violates (clustering,
  fat tails). So classical error formulas are overconfident → must earn error
  bars empirically.
- **Stationarity.** Most models assume the process doesn't change over time.
  Yours doesn't (regime-dependence) — the formal name for the most robust finding.

### Probability
- **Conditional probability.** `P(A|B)` = probability of A restricted to the world
  where B happened. Prediction IS conditioning — narrowing a distribution using
  information. Count-and-divide: `P(event|bin) = (cards in bin where event fired)
  / (cards in bin)` = **mean of the 0/1 event column within the bin** (that's why
  `groupby(bin).mean()` does it in one line).
- **Choosing the event chooses the moment.** Event = top-quintile return / vol /
  |return| probes the 1st / 2nd / 4th moment. Structural null: every event is
  "top 20%" → unconditional rate = 0.20 by construction.
- **Bayes' theorem** (introduced, not yet applied): prior + evidence → posterior;
  the math of updating belief and of confidence.

### Geometry / features
- **Vectors & feature space.** Each stock-day is a point in 5-D; a linear model is
  a *direction* through it, prediction = how far along that direction (a dot
  product).
- **Cross-sectional z-scoring.** Standardize each feature within each day →
  comparable scale AND market removed (absolute → peer-relative). Leakage-safe
  (uses only that day).
- **PCA (built from scratch).** Covariance → eigendecomposition → projection.
  Found the 5 features are **near-independent** (variance ≈ evenly spread across
  components; PC1 only ~28%, needs 4 PCs for ~88%). numpy matched sklearn to 1e-16.
- **Correlation vs mutual information.** Correlation sees only linear/monotone
  dependence; MI sees ANY. `MI = H(Y) − H(Y|X)` (uncertainty-before minus after),
  estimated via KSG nearest-neighbors (sklearn `mutual_info_regression`),
  meaningful only vs a **shuffle null**. Features with ~0 correlation but real MI
  carry non-linear (hump-shaped) info.

### Modelling
- **Linear regression two ways.** Model = `ŷ = w·f + b`. Solved exactly via the
  normal equation `w = (XᵀX)⁻¹Xᵀy` (numpy `lstsq`), AND iteratively via
  **gradient descent** (`grad = (2/n)Xᵀ(Xw−y)`, step `w ← w − lr·grad`). They
  converge to the same weights (matched to 1e-17) — proving gradient descent
  trustworthy for models with no formula (neural nets).
- **Squared error.** Fitting minimizes `Σ(ŷ−y)²`; punishes big misses, connects to
  projection geometry; sensitive to outliers (a real concern in fat tails).
- **R² vs IC — the key dissociation.** R² = variance explained (magnitude,
  market-contaminated); rank IC = did you order stocks right (within-day ranking).
  Proved they differ: residualizing changed R² but left IC **exactly** unchanged
  (subtracting a per-day constant can't reorder). Ranking is easier & more useful.
- **Rank IC / Spearman.** Cross-sectional rank correlation of predicted vs actual;
  robust to fat tails (uses order). The real scorecard.
- **Rolling validation + purge gap.** Train 3yr → test 1yr, roll forward; drop the
  boundary region so the training outcome can't leak into test features. Reveals
  stability vs regime-dependence directly (weights drifting across folds).
- **Residualization.** Subtract market (equal-wtd per-day mean of `fwd_ret_20d` —
  which IS the equal-weighted market return, by definition, not a shortcut) from
  the target. Lifts R² toward IC's implied level; leaves ranking untouched.

### Rigor / anti-self-deception
- **Block bootstrap.** For honest CIs on dependent data: resample whole time-blocks
  (21d) with replacement, recompute many (B=1000) times, read CI off the spread.
  By-block (not by-card) because ~1.8M cards ≠ 1.8M independent obs — effective n
  ≈ number of independent time-blocks. By-card resampling gives falsely tight CIs.
- **Shuffle nulls.** Destroy the relationship (shuffle), recompute many times;
  believe the real number only if it clears that noise floor.
- **Multiple testing / pre-registration.** Fix the set of tests in advance (no
  fishing); widen the significance bar with the number of tests (Bonferroni). At
  ~1.5M cards almost everything clears *statistical* significance → read **effect
  size and stability**, not just "CI excludes null."
- **Stability across sub-periods.** Recompute early half vs late half; trust only
  what appears in both. One-era-only = regime artifact.
- **Synthetic self-tests.** Before real data, every engine must recover a planted
  signal and reject known nulls on synthetic data. If it can't, it doesn't run on
  real data. (Done for every notebook.)

---

## Empirical findings (what the concepts produced)

### Feature space (exploration)
- The 5 features are **near-linearly-independent** (pale correlation heatmap, all
  |corr| < 0.22; blob-shaped PCA clouds; flat variance-explained). No two are the
  same axis. Momentum↔vol correlation ≈ +0.02 (near zero) — but recall the main
  project found them entangled via MI; the two are consistent (correlation =
  linear only, MI = nonlinear too).
- **Non-stationarity is visible:** log-turnover median drifts up 2011→2026 (but
  z-scoring differences it away each day); other features roughly stationary in
  level, bands widen in volatile years.
- **Shapes (conditional-mean plots):** momentum ≈ linear; realized-vol has an
  accelerating (concave) curve; **log-turnover and overnight-drift are inverted-U
  (hump) shaped** with ~0 correlation but real MI — nonlinear structure correlation
  misses. Reversal's shape is unstable (regime-dependent, early/late diverge).

### Linear regression (rolling folds, 2014-2025 test years)
- **Pooled rank IC ≈ +0.054, positive ~65% of days; R² ≈ −0.04 (absolute target).**
  The dissociation: near-zero/negative R² next to a real positive IC. Returns are
  near-unpredictable (IC ~0.05 is near the honest ceiling — returns depend on
  unknowable future news).
- **Weights across folds:** momentum weight **positive in all 12 folds** (robust);
  realized-vol **negative in most** (low-vol effect, intensifies in 2020-21);
  **reversal FLIPS SIGN** ~2019 (regime-dependent); turnover & overnight ≈ noise.
- **IC over time** waves above/below zero in long cycles (strong 2014/2019, weak
  2020-21 momentum-crash era, recovers 2025). Regime-dependence from a 2nd view.
- **V2 residualized target:** R² −0.042 → +0.0003; **rank IC EXACTLY unchanged**
  (+0.054) — proved per-day demeaning can't change within-day ranks. Market was
  28.6% of target variance. Weights essentially unchanged → feature findings are
  robust to target definition (real structure, not artifacts). (V3 official-index
  residualization pinned as low-priority: would also leave IC ~unchanged.)

### The moment investigation (conditional probability, THE headline)
Test: `P(event | trailing-252d feature-decile)` for 5 features × 3 events (R=return
1st moment, V=volatility 2nd, T=|return| 4th), block-bootstrap CIs, Bonferroni-
widened, early/late stability. Trailing-252d bins = point-in-time (bin edges from
prior 252 days only; costs the first ~year).

- **THE THESIS, PROVEN:** returns are near-unpredictable but **risk and tails are
  strongly and stably predictable.** Realized-vol → V event: P swings **0.037 →
  0.537** across deciles (~15x odds), monotone, razor-tight CIs, near-perfect
  early/late agreement. Realized-vol → T: 0.090 → 0.358.
- R-event gaps small (≤ +0.064 top-minus-bottom) for all features; V/T-event gaps
  large (up to +0.499).
- **U-shapes** in momentum/reversal/overnight for V and T events (both extremes
  predict turbulence) — vol itself is a monotone ramp (it measures agitation
  directly; the others only via magnitude, hence the U).
- Caveat learned: at ~1.5M cards, "CI excludes 0.20" fires for nearly everything →
  the honest discriminators are **effect size + early/late stability**, not the
  significance flag.

### Step 1 — directional tail (crash vs spike)
Split T into T-down (bottom-quintile return = crash) and T-up (top = spike).
- **The predictable tail structure is on the DOWNSIDE. Three features flag crash
  risk, each stable across eras:**
  - **Low momentum → crash:** P(crash)=0.292 vs P(spike)=0.205 at decile 0
    (asym +0.087). Strongest, most actionable. High momentum spike-tilted but
    weaker (0.249 vs 0.211).
  - **High realized-vol → crash:** 0.325 vs 0.228 (asym +0.097). Corrects the
    earlier loose "symmetric turbulence" read — high vol is **crash-skewed**.
  - **High overnight-drift → crash:** 0.260 vs 0.214 (asym +0.046). Unexpected,
    weakest, hold tentatively — flagged for idea generator ("does persistent
    overnight-up drift precede crashes, and why?").
- Reversal tails ≈ symmetric (genuine both-directional volatility). Log-turnover
  flat on both (weakest feature throughout).

**THE BOUNDARY held throughout:** predicting risk ≠ predicting returns ≠ making
money. A 15x vol-odds swing or a 29%-vs-20% crash probability is a real, stable
**risk** signal (position sizing, hedging, options) — NOT directional alpha, and
low-momentum-crashes is partly the flip side of the known weak momentum return.

---

## Code artifacts (all in moonshot_ml_sandbox/, notebooks + viz_exports_* folders)

- `..._v1_builder.ipynb` — builds the V1 dataset (THE deliverable
  `ml_sandbox_v1_dataset_<date>.parquet`).
- `..._v1_visualization.ipynb` — feature-space exploration (histograms,
  non-stationarity, correlation heatmap, PCA from scratch).
- `..._v1_regression.ipynb` — rolling linear regression, exact + gradient descent,
  R²/IC scoring, 6 teaching charts.
- `..._v2_residualized.ipynb` — residualized-target comparison (proves IC
  invariance).
- `..._nonlinearity_exploration.ipynb` — model-free: conditional-mean plots,
  corr-vs-MI-vs-null, conditional-IC interactions.
- `..._conditional_probability.ipynb` — the moment investigation (R/V/T events,
  trailing bins, block bootstrap). Requires reconstructing forward realized vol
  from the source panel.
- `..._step1_directional_tail.ipynb` — crash vs spike.

Every notebook: reads V1 dataset read-only, saves figures to its own
`viz_exports_*` folder via a `save_fig` helper (save BEFORE `plt.show()`, which
clears figures), prints a copy-paste summary block, and was validated on
synthetic planted-signal data before touching real data.

---

## Roadmap (locked plan of action)

Objectives: (a) learn probability & model-building concepts, (b) end with a real
direction for a volatility/movement model, (c) learn new modelling & pattern-
recognition techniques. Spine: **build toward a calibrated risk model, learning
each technique when the path needs it.**

**Phase 1 — finish characterizing the risk signal:**
- Step 1 — crash vs spike asymmetry. ✅ DONE (findings above).
- Step 2 — **transition probabilities (NEXT).** Volatility as a 2-state system
  (calm/turbulent); measure the 2×2 transition matrix `P(state next | state now)`.
  Concepts: states, Markov transitions, persistence. Diagonal = persistence;
  `P(T→T) ≫ 0.20` = clustering re-measured from the time axis. **Strategic payoff:
  a strong diagonal is the go/no-go signal for whether the regime-aware phase
  (Fork 2) is justified** — regimes are worth modelling only if states persist.
  Markov caveat: assumes next state depends only on current, not full history;
  testable later.

**Phase 2 — first real probabilistic model (core of objective b):**
- Step 3 — **logistic regression** for a risk event (P(high-vol) or P(crash) |
  features). Concepts: sigmoid squashing, maximum-likelihood fit, log-loss. The
  probabilistic sibling of the linear model; on-ramp to neural-net classifiers.
- Step 4 — **calibration** (reliability diagram). Ranking vs calibration are
  different: a model can rank right but its probabilities lie. Essential the
  moment a probability model exists.

**Phase 3 — flexibility & pattern recognition (objective c):**
- Step 5 — nonlinear risk model: bent features (targeted by exploration) then a
  small neural net. Concepts: bias-variance tradeoff *in practice*, activation
  functions. Learn nets on the signal that's actually there (risk).
- Step 6 — gradient boosting (tree ensemble, LightGBM-style). A different
  pattern-recognition mechanism; the practical quant workhorse; handles
  interactions automatically.

**Phase 4 — regime-aware (Fork 2), armed:**
- Step 7 — define regimes leak-free, re-run model-free exploration WITHIN each
  regime, then model within regimes. Highest overfitting risk (two flexibility
  layers on less data) → done last, with sharpest instincts. Justified only if
  Step 2 shows states persist.

**Discipline for the whole roadmap:** let findings redirect the plan (it's a
strong hypothesis, not a script); each phase is a complete destination on its own;
hold the risk ≠ return ≠ money boundary always.

**Pinned / parked:** V3 official-Nifty-500 residualization (low priority — won't
move IC); the two-feature joint-conditioning interaction probe (idea 2);
feature-event MI capstone (idea 5, tidy summary, redundant for discovery);
absolute-threshold vol event (where the 2018-vs-2020 context argument bites).

---

## Non-negotiable rules (carried from moonshot_compounding)
- Reads the frozen panel READ-ONLY; writes only into `moonshot_ml_sandbox/`.
- Timestamped outputs, never overwritten.
- NaN = unknown, never interpolated/zero-filled; a non-existent date is deleted.
- Every engine synthetic-self-tested (recover planted signal, reject nulls) before
  real data.
- Leakage wall inherited from the dataset build — the test can't fix a leak, only
  inherit a clean one; hence the heavy dataset-level validation.
- Hold the risk ≠ return ≠ money boundary in every interpretation.
