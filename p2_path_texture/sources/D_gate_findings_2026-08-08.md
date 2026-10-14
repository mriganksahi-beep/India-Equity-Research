# Test D — Wavelet Scale-Energy — D-GATE — FINDINGS (SHELVED)

id: microstructure_D_wavelet_gate
date: 2026-08-08
mechanism: Test D gate (Microstructure & Anomaly batch) — gated, variance-ratio-first
hypothesis: >
  The balance of slow-scale (trending) vs fast-scale (mean-reverting) movement
  energy in a stock's recent path predicts forward returns, independent of vol.
  Because total wavelet energy = realized variance (mechanical identity), raw band
  energy IS vol in disguise; only normalized coarse-band SHARES can survive. The
  variance ratio asks the identical economic question, is intrinsically
  vol-normalized, and skips wavelet short-series edge-effect fragility — so the GATE
  runs on the variance ratio; wavelet code is built ONLY if the VR gate clears.
category: microstructure / path-structure / scale-energy / gate
mechanism_tags: variance_ratio, wavelet_energy, scale_structure, vol_orthogonalization,
  short_series_fragility, path_texture, gated_testing

data_and_universe: >
  Nifty 500 PIT panel, v3. TRAINING 2010-2021 ONLY (hold-out walled off). 59,184
  stock-months, 865 tickers. 8 corp-action tickers screened. VR over the [t-252,t-21]
  formation window. VR(q) = Var(q-day)/(q*Var(1-day)), overlapping Lo-MacKinlay,
  q=21 primary. Gate order: (1) vol-orthogonalization FM, (2) dispersion, (3)
  block-bootstrap rank stability.

estimator_verification: >
  The VR estimator was verified against known cases BEFORE building, and TWO bugs
  were caught and fixed in the process (a normalization error in the overlapping
  estimator; then confirmation that non-overlapping is too block-starved at 252 obs).
  Final estimator: textbook-correct at N=1000 (trend 1.87, mean-rev 0.40, RW 1.03),
  provably vol-invariant (ratio of variances). CRITICAL SHORT-SERIES FINDING: at
  N=252 the estimator is UNBIASED (RW mean 0.95 across 200 seeds) but carries HIGH
  per-stock noise (std 0.30, range 0.32-1.91). This foreshadowed the null: real NSE
  stocks are near-random-walk, so per-stock VR is dominated by ~0.30 estimation noise.

result: >
  DECISION: SHELVE D at gate 1. No wavelet code built (variance-ratio-first paid off).

  GATE 1 (vol-orthogonalization):
    VR coef UNCONTROLLED:            -0.00059  (NW t=-0.41)
    VR-residual +RV+mom+Amihud:      -0.00028  (NW t=-0.23)
  The gate is labeled "orthogonalization fail" but the honest read (as with M7) is
  STRONGER: the UNCONTROLLED t of -0.41 means VR had NO forward-return power to begin
  with — there was no signal for the vol control to remove. Not "vol in disguise";
  "no signal at all."
  GATE 2 (dispersion): not reached (gate 1 failed).
  GATE 3 (stability): not reached (gate 1 failed).

  So the wavelet localization is NOT built: the variance ratio is the clean, robust,
  edge-effect-free proxy for coarse-band wavelet energy; if it shows nothing, the
  fragile edge-effect-prone wavelet version cannot rescue it.

validation_summary: >
  hold-out wall: gate on 2010-2021 only; hold-out untouched. PASS.
  estimator: pre-verified (trend/mean-rev/RW separation at N=1000; vol-invariance;
    short-series noise probe). PASS.
  coverage: VR(q=21) 100% of the VR panel. PASS.
  gate order: vol-orthogonalization first, short-circuits on failure. PASS.

holdout_status: n/a (shelved on training-side gate 1; hold-out not consulted)

decision: SHELVE (variance ratio carries no vol-independent forward-return signal;
  wavelet localization not built)

insight: >
  The balance of slow- vs fast-scale movement energy in a stock's recent path does
  NOT predict forward returns on NSE (uncontrolled t=-0.41, before any vol control).
  Combined with M7 (entropy), a coherent structural fact now holds: the fine-grained
  TEXTURE of a stock's recent return path — its disorder (entropy) AND its
  scale-energy balance (variance ratio / wavelets) — is not a usable cross-sectional
  return predictor at daily frequency / 252-obs windows. Momentum (the MAGNITUDE of
  the path) works; the path's TEXTURE does not. The variance-ratio-first design
  killed a wavelet hypothesis for a few Var() calls instead of a MODWT/Daubechies
  pipeline, and reached the exact "dead" verdict the spec anticipated.

incidental_observations: >
  (1) This is the SECOND path-texture signal (after M7 entropy) to die the same way:
  no cross-sectional return signal, ultimately traceable to near-random-walk NSE
  daily returns being unmeasurable in shape at 252 obs. The pattern is now strong
  enough to treat as a structural prior, not two coincidences. (2) The short-series
  noise probe (RW VR std 0.30 at 252 obs) is a reusable diagnostic: any path-shape
  estimator on a 250-day window should be noise-probed on random-walk input before
  trusting cross-sectional differences. (3) NULL is on cross-sectional RETURN
  prediction only — scale structure could still matter for risk/crash character
  (Test A's domain), a different question D does not address.

flag_for_idea_generator: >
  STRUCTURAL PRIOR to carry forward: path-TEXTURE signals (entropy, scale-energy,
  variance ratio, and by extension wavelets) do not predict cross-sectional returns
  on NSE daily data — near-random-walk returns at 252 obs make the texture both
  noise-dominated per-stock and non-predictive. Before proposing any further
  path-shape return signal, confront this prior; the burden is on showing why a new
  shape measure would escape what killed entropy AND variance ratio. Lower-frequency
  (weekly/monthly) path structure remains formally untested and is the only obvious
  escape hatch. Path texture for RISK (not return) is a separate, still-open question.

related_ids: >
  momentum_enh_M7_path_entropy (the first path-texture null; same near-random-walk
  root cause). momentum_enh_M3_volscaled (the vol-in-disguise failure mode the gate's
  orthogonalization guards against). The wavelet green-field idea routed here from M7
  is now closed at the variance-ratio stage.

running_tally: >
  Tally 32 (D-gate = trial 32). Gates 2-3 never executed (gated off), so they are NOT
  trials. D shelved without a build or wavelet code. Feeds the shared-hold-out DSR
  denominator at reveal.
