# moonshot_ml_sandbox — Step 2 Record: Volatility-State Persistence (Markov Analysis)

**What this document is.** A complete, self-contained record of the Step 2
transition/persistence analysis — written so future-you can reconstruct *why each
calculation was done, what it means intuitively, and how it was computed*, not
just the results. Companion to the main build log. Read this before designing
Fork 2 (regime-aware modelling).

**The one-line result:** volatility regimes are **real, strong, stable, extreme-
concentrated, and duration-dependent** → Fork 2 is JUSTIFIED (GO), and should be
built **duration-aware** and **extreme-focused**.

---

## 0. Why we ran this at all (the motivation)

The regression tests showed the feature→return relationship *oscillates over time*
(weights drift across folds, reversal flips sign, IC waves above/below zero). That
is **regime-dependence**, and it's the premise of the whole regime-aware phase
(Fork 2). But regime-aware modelling is only worth doing **if states actually
persist** — if this month's state predicts next month's. If states reset randomly
each month, "regime" is a meaningless label and Fork 2 would be building on sand.

So Step 2 is the **go/no-go test**: measure whether volatility (as a state a stock
occupies) persists enough to justify Fork 2. We tested volatility specifically
(Layer 1) because it's data-rich; the parallel question "does each feature's
*predictiveness* persist" (Layer 2, market-level, noisier) is parked for later.

---

## 1. Building the states (what we measured the persistence OF)

**The idea.** Treat each stock's volatility as a *state* it occupies each month and
moves between. Two-state version: **calm** vs **turbulent**. Five-state version:
volatility quintiles Q0 (calmest) .. Q4 (most turbulent).

**How computed.** For each calendar month, rank every stock's forward realized
volatility *cross-sectionally within that month*; top 20% = turbulent (state 1),
rest = calm (state 0). Keep **one observation per stock per month** (the last card
in the month). Each stock becomes an ordered monthly string of states:
`calm, calm, turbulent, turbulent, calm, ...`

**Two design choices, and why they matter:**
- **Cross-sectional-within-month ranking:** "turbulent" means top-quintile *relative
  to peers that month*, so the state means the same thing in a calm year (2017) and
  a wild one (2020). Avoids a fixed vol threshold that 2020 would saturate.
- **Monthly, non-overlapping sampling:** consecutive states are one non-overlapping
  ~21-day vol-window apart. Sampling daily (overlapping windows) would
  **manufacture fake persistence**; monthly is the clean choice. This was THE
  construction to get right.

**Result:** unconditional P(turbulent) = **0.2009** ≈ 0.20 — exactly as it must be,
since turbulent = top quintile by construction. (First confirmation the states are
built right.)

---

## 2. The transition matrix (the core calculation)

**The idea.** Collect every one-step move probability into a grid. Entry P_ij =
P(next state = j | current state = i). Read the *row* as "from state i," the
*column* as "to state j."

**How computed — the estimator P̂_ij = N_ij / N_i:**
Walk each stock's monthly string. Every time two observations are the **same stock
in CONSECUTIVE months** (month numbers differ by exactly 1 — never across a gap,
never across stocks), tally a transition from the first state to the second into a
count grid N. Then divide each row by that row's total. This is the same
count-and-divide used throughout, now with condition and event being the *same
variable one step apart in time*.

**Point-in-time discipline:** a transition is legitimate only if state-now and
state-next are genuinely consecutive months for one stock. Gaps (suspended stock,
missing month) are NOT counted as transitions — the pair straddling a gap isn't a
true one-month move.

**Result (2-state):**
```
                → calm     → turbulent
  calm now      0.8562      0.1438
  turbulent     0.5741      0.4259
```
**Reading:** P(turbulent→turbulent) = **0.4259** — a turbulent stock is >2x as
likely to stay turbulent (0.43) as a random stock is to be turbulent (0.20 baseline).
P(calm→calm) = **0.8562** — calm is even stickier. Both diagonals well above
baseline = the signature of real memory.

**Validity check:** each row sums to 1 (0.8562+0.1438 = 1; 0.5741+0.4259 = 1).
A row-stochastic matrix; confirms the counting has no leak.

---

## 3. The persistence scalar λ (the headline number)

**The idea.** Collapse the two diagonal entries into ONE interpretable number for
"how sticky is the system." It runs like a correlation: −1 (always alternates),
0 (memoryless/independent), +1 (frozen, never switches).

**How computed — λ = P_TT + P_CC − 1:**
`λ = 0.4259 + 0.8562 − 1 = +0.2821`

**Why this formula IS the persistence measure (the intuition):** if the state were
independent month-to-month, both matrix rows would be *identical* (the "from" state
wouldn't matter), giving P_TT = 0.20, P_CC = 0.80, so λ = 0.20 + 0.80 − 1 = 0.
Deviation from identical rows = memory = λ ≠ 0. λ is also (a) the month-to-month
autocorrelation of the state indicator, and (b) the second eigenvalue of P — the
same object seen three ways.

**How we trusted it — the block bootstrap:** resample whole **12-month blocks** of
history with replacement, recompute the full matrix and λ on each resample, 1000
times, take the 2.5th–97.5th percentiles. Block (not per-stock-month) resampling is
essential: it treats the ~180 months as ~a dozen independent year-blocks rather than
pretending every stock-month is independent, so the CI reflects real information
content (dependent data → wider, honest bars).

**Result:** λ = **+0.2821, CI [+0.2659, +0.2989]**. Tight, clearly excludes 0 →
persistence is REAL and substantial (not noise).

---

## 4. The stationary distribution (the correctness tripwire)

**The idea.** The long-run fraction of time the system spends in each state — the
distribution that reproduces itself under one step. Push any starting distribution
forward with P repeatedly and it settles here, regardless of where it started.

**How computed — solve π* P = π*:** this says π* is the **left eigenvector of P with
eigenvalue 1**. Computed via eigen-decomposition of Pᵀ (pick eigenvalue nearest 1,
normalize to sum 1). Cross-checked by iterating (multiply a start distribution by P
200 times) and confirming it lands on the same point.

**Why eigenvalue 1 always exists:** rows of P sum to 1 → the all-ones column vector
is a right eigenvector with eigenvalue 1 → a left eigenvector with eigenvalue 1
(= π*) must exist too. Every valid transition matrix has a stationary distribution,
structurally.

**Result:** π* = [calm **0.7996**, turbulent **0.2004**].
**Why this is the tripwire:** turbulent = top-quintile by construction, so ~20% of
stock-months are turbulent long-run, so π*_turbulent MUST come out ≈ 0.20. It came
out 0.2004 — four-decimal agreement. This is the single strongest reason to trust
every other number: if the counting leaked anywhere, this check would have failed.

---

## 5. Multi-step propagation (memory fade — how long the state "remembers")

**The idea.** Given turbulent now, what's P(turbulent) in n months? As n grows, the
elevated odds should decay toward the 0.20 baseline (memory fades).

**How computed — matrix powers Pⁿ:** the (turbulent, turbulent) entry of Pⁿ.
Matrix multiplication automatically sums over all intermediate time-paths — Pⁿ[T,T]
= sum over every possible sequence of intermediate states, each weighted by its
probability. (Verified by hand earlier: P²[T,T] = P_TC·P_CT + P_TT·P_TT.)

**Result:** P(turbulent | turbulent now) over n = [1, 2, 3, 6, 12, 24] =
**[0.426, 0.264, 0.218, 0.201, 0.200, 0.200]**. Decays to the 0.20 stationary
baseline by ~month 6. **Meaning:** being turbulent now elevates your odds for about
half a year, then washes out.

---

## 6. The 5-state matrix (graded persistence — texture)

**The idea.** Two states is coarse. Do *extreme* volatility states persist more than
mild ones? And do stocks drift gradually or jump?

**How computed:** identical estimator, states = vol quintiles Q0..Q4.

**Result — diagonal (stay-probabilities):** [0.438, 0.260, 0.243, 0.263, **0.426**].
```
5-state matrix (rows=from, cols=to):
        Q0     Q1     Q2     Q3     Q4
Q0    0.438  0.246  0.156  0.099  0.061
Q1    0.255  0.260  0.216  0.162  0.108
Q2    0.161  0.224  0.243  0.215  0.158
Q3    0.096  0.169  0.224  0.263  0.248
Q4    0.054  0.099  0.161  0.260  0.426
```
**Two readings:**
- **Extremes are stickiest:** Q0 (0.438) and Q4 (0.426) stay-probabilities are ~2x
  baseline; the middle quintiles (~0.24–0.26) are near baseline (transient). So the
  calmest and wildest states are the stable *regimes*; the middle is a transition
  zone stocks pass through.
- **Stocks drift one quintile at a time:** off-diagonal mass sits NEAR the diagonal
  (Q3→Q4 = 0.248, Q4→Q3 = 0.260), while far corners are tiny (Q0→Q4 = 0.061,
  Q4→Q0 = 0.054). Volatility moves gradually, not in jumps.

---

## 7. The duration-dependence (Markov) test — THE key structural finding

**The idea.** The single matrix assumes the **first-order Markov property**: next
state depends ONLY on current state, not on history. Test it: does P(turbulent→
turbulent) depend on *how long* the stock has already been turbulent? Under Markov
it must be FLAT (only current state matters). A RISING line = memory beyond one step.

**How computed:** track a running duration counter per stock (how many consecutive
months already turbulent); bucket each turbulent→next transition by that counter;
compute P(turbulent next) within each bucket.

**Result:**
```
months already turbulent:  0     1     2     3     4     5     6+
P(turbulent next):        0.349 0.454 0.555 0.604 0.638 0.674 0.699
```
Range 0.35 — far above the 0.08 flatness threshold. **First-order Markov is
VIOLATED.** A stock that JUST turned turbulent has only 35% odds of staying; one
turbulent 6+ months has 70%. **Turbulence deepens the longer it lasts** — positive
duration dependence. The single matrix (0.426) averages these together, understating
persistence for long-turbulent names and overstating it for just-turned ones.
**Economic sense:** a just-spiked stock may be a one-off (earnings); a stock volatile
for half a year is in a genuine distressed regime that self-perpetuates.

**Caution held (why to trust the existence, hold the magnitude loosely):** stocks
turbulent 6+ months are a *selected* subset (the ones that didn't calm down), so some
of the rise could be survivorship/selection rather than pure duration dependence. But
the effect is so large and monotone that genuine duration dependence is almost
certainly present too. Trust the direction/existence strongly; hold exact magnitude
loosely. Does NOT weaken the GO — it sharpens Fork 2's design.

---

## 8. Stability across eras (regime-artifact check)

**How computed:** re-ran the whole 2-state estimation separately on the early and
late halves of history. **Result:** early λ = +0.290, late λ = +0.274. Nearly
identical → persistence is structural, present in both halves, NOT a one-era artifact.

---

## 9. Why we trust the whole thing (evidence stack, strongest first)

1. **Stationary check passed to 4 decimals** (π*_turb = 0.2004 vs 0.20 baseline) —
   a hard correctness tripwire that would have failed on any counting leak.
2. **λ CI tight and excludes 0** [+0.266, +0.299], with honest block-bootstrap bars.
3. **Early/late λ agree** (+0.290 / +0.274) — structural, not a fluke era.
4. **Engine self-tested** before real data: recovered a known planted matrix, λ,
   stationary; correctly distinguished Markov from non-Markov synthetic chains.
5. **Economically sensible** — vol clustering + deepening distress are among the most
   robust facts in empirical finance; a strong clean result is expected.

The one caveat: duration-test magnitude may be inflated by survivorship (§7).

---

## 10. What it means for the future (prediction + roadmap)

**About prediction:** a stock's current volatility state genuinely forecasts its
near-future state — ~43% odds of turbulence next month vs 20% baseline, rising to
~70% if turbulent a while. Forecastable structure, usable for risk management
(sizing, hedging). The "risk is predictable" thesis, re-confirmed from the time axis.

**About the roadmap — Fork 2 GO, with sharpened design:**
- **JUSTIFIED:** regimes are real, strong (λ=0.28), stable across eras.
- **Build it DURATION-AWARE:** track how long a stock has been in a state, not just
  the current state — the single matrix understates persistence for long-turbulent
  names (§7). A first-order model leaves structure on the table.
- **Build it EXTREME-FOCUSED:** the calm and wildly-turbulent states are the stable,
  meaningful regimes; the middle quintiles are transitional (§6).
- **Volatility drifts gradually** (one quintile at a time), so regime transitions are
  smooth, not abrupt — relevant for any model acting on the state.

We bought the right architecture for Fork 2 *before* building it — the purpose of
running this test first.

---

## Code artifact
`moonshot_ml_sandbox_step2_volatility_markov.ipynb` — mount → reconstruct forward
vol → build monthly states → 2-state matrix + λ + bootstrap CIs → stationary +
memory-fade → 5-state matrix → duration test → early/late stability → summary.
Reads V1 dataset + source panel READ-ONLY; figures to `viz_exports_step2/`.
Synthetic-self-tested (recovers planted matrix/λ/stationary, distinguishes
Markov from non-Markov) before touching real data.
