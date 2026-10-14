"""
conditional_mi.py
=================
Conditional mutual information for the moonshot MI screen (quant-eye analysis 1).

QUESTION
--------
How much does feature A tell us about forward returns OVER AND ABOVE what feature
B already tells us?  Formally  I(A ; forward_return | B), per rebalance date,
cross-sectionally, on ranks.

ESTIMATOR
---------
npeet_plus.cmi -- a KSG-style k-NN conditional-MI estimator.  VALIDATED before use
against three known-answer cases (see conditional_mi_selftest.py):
  * independent A,B,Y            -> CMI ~ 0        (no false additivity)
  * A,B both noisy copies of the
    same return-driver (redundant) -> CMI drops toward 0  (detects redundancy)
  * A,B carry separate signal     -> CMI stays large      (preserves additivity)

NULL AND CI  (same discipline as the base screen)
-------------------------------------------------
Per date:  raw_cmi  = cmi(A_rank, ret_rank | B_rank)
           null_cmi = mean over within-date shuffles of the RETURN ranks
           cmi_excess(date) = raw_cmi - null_cmi
Point estimate = mean of per-date cmi_excess.  CI = block bootstrap over dates
(the independent unit).  A near-zero excess => A is REDUNDANT given B.  A clearly
positive excess (CI excludes 0) => A carries information B does not.

Estimation is 3-variable k-NN, noisier than the 1-D base screen, so CIs are wider
BY DESIGN -- the honest cost of a fully non-parametric conditional estimate (no
linearity assumption).  We do NOT tighten them by assuming a functional form.

INTERFACE
---------
Reuses mi_screen's Panel, feature builders, forward_return, rebalance grid,
_rank01, PIT eligibility, and hold-out guard.  Pairs are specified as
(A_name, B_name): "A given B".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from npeet_plus import cmi, mi as npeet_mi

import mi_screen as M


def _col(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _eligible_triple(panel, fa_wide, fb_wide, fwd_wide, t, h):
    """Return rank-transformed (A, B, forward_return) for the eligible cross
    section on date t, or None.  Eligibility: flag==1 AND all three present AND
    forward window does not cross the hold-out wall (reuses mi_screen guards)."""
    i = panel.dates.get_indexer([t])[0]
    if i < 0 or i + h >= len(panel.dates):
        return None
    if panel.dates[i + h] > M.TRAIN_END:
        return None  # hold-out guard
    flag = panel.flag.loc[t]
    a = fa_wide.loc[t]; b = fb_wide.loc[t]; y = fwd_wide.loc[t]
    elig = (flag == 1) & a.notna() & b.notna() & y.notna()
    n = int(elig.sum())
    if n < M.MIN_STOCKS_PER_DATE:
        return None
    return (M._rank01(a[elig].to_numpy(float)),
            M._rank01(b[elig].to_numpy(float)),
            M._rank01(y[elig].to_numpy(float)), n)


def per_date_cmi_excess(triples, rng, n_shuffle=M.N_SHUFFLE):
    """One conditional-MI-excess per date: raw CMI minus shuffled-return null."""
    out = np.empty(len(triples))
    for j, (a, b, y, n) in enumerate(triples):
        raw = cmi(_col(a), _col(y), _col(b), k=M.KNN_K)
        nulls = np.empty(n_shuffle)
        for s in range(n_shuffle):
            nulls[s] = cmi(_col(a), _col(y[rng.permutation(n)]), _col(b), k=M.KNN_K)
        out[j] = raw - nulls.mean()
    return out


def conditional_mi_pair(panel, a_name, b_name, horizon_name,
                        rng_seed=M.RNG_SEED, n_boot=M.N_BOOT, verbose=True):
    """condMI_excess for 'A given B' at one horizon, with block-bootstrap CI.
    Also returns the MARGINAL MI_excess of A (from the base screen logic) so the
    reader can see how much of A's information SURVIVES conditioning on B."""
    rng = np.random.default_rng(rng_seed)
    h = M.HORIZONS[horizon_name]
    fa = M.FEATURES[a_name](panel)
    fb = M.FEATURES[b_name](panel)
    fwd = M.forward_return(panel, h)
    rebal = M.month_end_rebalance_dates(panel)

    triples, dropped = [], 0
    for t in rebal:
        tr = _eligible_triple(panel, fa, fb, fwd, t, h)
        if tr is None:
            dropped += 1
        else:
            triples.append(tr)
    if len(triples) < 3:
        return {"pair": f"{a_name}|{b_name}", "horizon": horizon_name,
                "n_dates": len(triples), "note": "too few dates"}

    pde = per_date_cmi_excess(triples, rng)
    m = len(pde)
    boot = np.array([pde[rng.integers(0, m, m)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, M.CI_LOW), np.percentile(boot, M.CI_HIGH)

    # marginal MI_excess of A alone, same dates, for the survival comparison
    css_a = [M.DateCrossSection(date=None, feat_rank=a, fwd_rank=y, n=n)
             for (a, b, y, n) in triples]
    pde_marg = M.per_date_mi_excess(css_a, np.random.default_rng(rng_seed))
    marg = float(pde_marg.mean())

    excess = float(pde.mean())
    survived = excess / marg if marg > 1e-9 else np.nan
    res = {"pair": f"{a_name}|{b_name}", "horizon": horizon_name,
           "n_dates": m, "n_dropped": dropped,
           "marginal_mi_excess": marg,
           "cond_mi_excess": excess, "ci_low": float(lo), "ci_high": float(hi),
           "sig": bool(lo > 0), "fraction_survived": survived}
    if verbose:
        star = "*" if lo > 0 else " "
        print(f"  {a_name:>16} | {b_name:<15} @{horizon_name}: "
              f"marg={marg:+.4f} -> cond={excess:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] {star} "
              f"survived={survived:.0%} (n={m})")
    return res


def run_conditional_pairs(panel, pairs, horizon_name="1m",
                          rng_seed=M.RNG_SEED, verbose=True):
    """pairs: list of (A_name, B_name). Runs condMI(A|B) for each at one horizon."""
    return [conditional_mi_pair(panel, a, b, horizon_name, rng_seed, verbose=verbose)
            for (a, b) in pairs]


# ===========================================================================
# DIAGNOSTIC CHECKS on the conditional-MI estimator (2026-08-10)
# ---------------------------------------------------------------------------
# The real-pair run returned >100% survival on ALL four pairs (uniform synergy),
# which is suspicious. These checks test the ESTIMATOR, not the market -- no DSR
# cost. They bracket its behaviour:
#   * condition on NOISE  -> survival should be ~100% (noise conditions nothing)
#   * condition on SELF   -> survival should be ~0%   (a feature adds nothing to itself)
#   * marginal two ways   -> the two estimators (sklearn KSG vs npeet) must agree
#                            on the same rank inputs, or the "survived" ratio is
#                            comparing incompatible scales.
# If noise-conditioning inflates >100% or self-conditioning fails to collapse,
# the uniform synergy is an estimator artifact, not a market finding.
# ===========================================================================

def _collect_triples_with_conditioner(panel, fa_wide, fwd_wide, h, cond_kind,
                                      rng, b_wide=None):
    """Build per-date (A_rank, C_rank, y_rank, n) where C is the conditioner.
    cond_kind: 'noise' (random per date), 'self' (copy of A), or 'feature'
    (use b_wide). Same PIT eligibility + hold-out guard as the real pipeline."""
    rebal = M.month_end_rebalance_dates(panel)
    triples = []
    for t in rebal:
        i = panel.dates.get_indexer([t])[0]
        if i < 0 or i + h >= len(panel.dates):
            continue
        if panel.dates[i + h] > M.TRAIN_END:
            continue
        flag = panel.flag.loc[t]
        a = fa_wide.loc[t]; y = fwd_wide.loc[t]
        base_elig = (flag == 1) & a.notna() & y.notna()
        if cond_kind == "feature":
            base_elig = base_elig & b_wide.loc[t].notna()
        n = int(base_elig.sum())
        if n < M.MIN_STOCKS_PER_DATE:
            continue
        ar = M._rank01(a[base_elig].to_numpy(float))
        yr = M._rank01(y[base_elig].to_numpy(float))
        if cond_kind == "noise":
            cr = M._rank01(rng.standard_normal(n))          # pure noise conditioner
        elif cond_kind == "self":
            cr = ar.copy()                                   # condition on itself
        elif cond_kind == "feature":
            cr = M._rank01(b_wide.loc[t][base_elig].to_numpy(float))
        else:
            raise ValueError(cond_kind)
        triples.append((ar, cr, yr, n))
    return triples


def diagnostic_conditioning(panel, feature_names, horizon_name="1m",
                            rng_seed=M.RNG_SEED, verbose=True):
    """For each feature: marginal MI_excess (BOTH estimators), survival when
    conditioned on NOISE, and survival when conditioned on SELF. One row per
    feature so the artifact (if any) is obvious on one screen."""
    h = M.HORIZONS[horizon_name]
    fwd = M.forward_return(panel, h)
    rows = []
    for fname in feature_names:
        rng = np.random.default_rng(rng_seed)
        fa = M.FEATURES[fname](panel)

        # --- marginal two ways, on identical rank inputs (Check 3) ---
        tri_noise = _collect_triples_with_conditioner(panel, fa, fwd, h, "noise", rng)
        if len(tri_noise) < 3:
            continue
        # marginal via sklearn KSG (base screen) and via npeet mi, same inputs
        sk_marg, np_marg = [], []
        for (a, c, y, n) in tri_noise:
            sk_marg.append(M._ksg_mi(a, y, rng))
            np_marg.append(float(npeet_mi(_col(a), _col(y), k=M.KNN_K)))
        sk_marg = float(np.mean(sk_marg)); np_marg = float(np.mean(np_marg))

        # --- condition on NOISE (Check 1): survival should be ~100% ---
        pde_noise = per_date_cmi_excess(tri_noise, np.random.default_rng(rng_seed),
                                        n_shuffle=M.N_SHUFFLE)
        cond_noise = float(pde_noise.mean())

        # --- condition on SELF (Check 2): survival should collapse to ~0% ---
        rng2 = np.random.default_rng(rng_seed)
        tri_self = _collect_triples_with_conditioner(panel, fa, fwd, h, "self", rng2)
        pde_self = per_date_cmi_excess(tri_self, np.random.default_rng(rng_seed),
                                       n_shuffle=M.N_SHUFFLE)
        cond_self = float(pde_self.mean())

        surv_noise = cond_noise / sk_marg if abs(sk_marg) > 1e-9 else np.nan
        surv_self = cond_self / sk_marg if abs(sk_marg) > 1e-9 else np.nan
        rows.append(dict(feature=fname, n_dates=len(tri_noise),
                         marg_sklearn=sk_marg, marg_npeet=np_marg,
                         marg_ratio=np_marg / sk_marg if abs(sk_marg) > 1e-9 else np.nan,
                         cond_on_noise=cond_noise, survived_noise=surv_noise,
                         cond_on_self=cond_self, survived_self=surv_self))
        if verbose:
            print(f"  {fname:>16}: marg(sklearn)={sk_marg:+.4f} marg(npeet)={np_marg:+.4f} "
                  f"ratio={np_marg/sk_marg if abs(sk_marg)>1e-9 else float('nan'):.2f} | "
                  f"cond|noise survived={surv_noise:.0%} | cond|self survived={surv_self:.0%}")
    return pd.DataFrame(rows)


# ===========================================================================
# CORRECTED CONDITIONAL-MI METRIC  (2026-09 correction run)
# ---------------------------------------------------------------------------
# The first conditional run reported survival as (npeet cond / sklearn marginal),
# mixing two differently-scaled estimators, and judged it against a naive 100%.
# Diagnostics showed the noise floor is feature-specific (44%-113%) and the
# marginal ratio ranges 0.68-1.42. So the fix:
#
#   1. Use ONE estimator (npeet) for BOTH the conditional and the marginal.
#   2. Judge each pair against its OWN noise floor: condition A on several random
#      noise variables, take the mean conditional-MI-excess as the floor.
#   3. Report the ABSOLUTE excess above floor:  cond_on_B_excess - noise_floor,
#      in nats, with a block-bootstrap CI over dates. CI excludes 0 => B leaves
#      A with genuine information beyond what conditioning on nothing would show,
#      i.e. real ADDITIVITY. (Ratios blew up before; absolute is robust.)
#
# npeet is used throughout, so there is no cross-estimator scale mismatch.
# ===========================================================================

def _npeet_cmi_excess_per_date(triples, rng, n_shuffle=M.N_SHUFFLE):
    """Per-date conditional-MI-excess using npeet for BOTH raw and null.
    triples: list of (A_rank, C_rank, y_rank, n). Returns one excess per date."""
    out = np.empty(len(triples))
    for j, (a, c, y, n) in enumerate(triples):
        raw = cmi(_col(a), _col(y), _col(c), k=M.KNN_K)
        nulls = np.empty(n_shuffle)
        for s in range(n_shuffle):
            nulls[s] = cmi(_col(a), _col(y[rng.permutation(n)]), _col(c), k=M.KNN_K)
        out[j] = raw - nulls.mean()
    return out


def corrected_pair(panel, a_name, b_name, horizon_name="1m",
                   n_noise_draws=10, rng_seed=M.RNG_SEED, n_boot=M.N_BOOT,
                   verbose=True):
    """Corrected additivity test for 'A given B'.

    Returns absolute excess of (A conditioned on B) above (A conditioned on noise),
    both computed with npeet on identical eligible cross-sections, with a
    block-bootstrap CI over dates. A stable multi-draw noise floor is used.
    """
    rng = np.random.default_rng(rng_seed)
    h = M.HORIZONS[horizon_name]
    fa = M.FEATURES[a_name](panel)
    fb = M.FEATURES[b_name](panel)
    fwd = M.forward_return(panel, h)
    rebal = M.month_end_rebalance_dates(panel)

    # Build the ALIGNED eligible set once: dates where A, B, y all present and
    # PIT/hold-out valid. Noise conditioning uses the SAME rows, so floor and
    # real are strictly comparable (same dates, same stocks, same n per date).
    triples_B, dropped = [], 0
    for t in rebal:
        tr = _eligible_triple(panel, fa, fb, fwd, t, h)  # (A, B, y, n)
        if tr is None:
            dropped += 1
        else:
            triples_B.append(tr)
    if len(triples_B) < 3:
        return {"pair": f"{a_name}|{b_name}", "horizon": horizon_name,
                "n_dates": len(triples_B), "note": "too few dates"}

    # (1) real: condition A on B, npeet throughout
    pde_B = _npeet_cmi_excess_per_date(triples_B, rng)

    # (2) marginal of A (npeet), same rows, for context
    marg = np.mean([float(npeet_mi(_col(a), _col(y), k=M.KNN_K))
                    for (a, b, y, n) in triples_B])

    # (3) stable noise floor: condition A on n_noise_draws independent noise vars,
    #     reusing the SAME (A, y, n) per date; average the per-date excess across
    #     draws, giving one floor series aligned to the same dates.
    floor_draws = np.empty((n_noise_draws, len(triples_B)))
    for d in range(n_noise_draws):
        tri_noise = [(a, M._rank01(rng.standard_normal(n)), y, n)
                     for (a, b, y, n) in triples_B]
        floor_draws[d] = _npeet_cmi_excess_per_date(tri_noise, rng,
                                                    n_shuffle=max(5, M.N_SHUFFLE // 2))
    floor_per_date = floor_draws.mean(axis=0)   # stable floor, per date

    # (4) corrected metric per date: (A|B excess) - (A|noise floor), then bootstrap
    diff = pde_B - floor_per_date
    m = len(diff)
    boot = np.array([diff[rng.integers(0, m, m)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, M.CI_LOW), np.percentile(boot, M.CI_HIGH)

    res = {"pair": f"{a_name}|{b_name}", "horizon": horizon_name, "n_dates": m,
           "n_dropped": dropped, "marginal_mi_npeet": float(marg),
           "cond_on_B_excess": float(pde_B.mean()),
           "noise_floor": float(floor_per_date.mean()),
           "floor_spread": float(floor_draws.mean(axis=1).std()),  # across-draw stability
           "excess_above_floor": float(diff.mean()),
           "ci_low": float(lo), "ci_high": float(hi),
           "additive": bool(lo > 0)}
    if verbose:
        star = "*" if lo > 0 else " "
        print(f"  {a_name:>16} | {b_name:<15} @{horizon_name}: "
              f"cond={res['cond_on_B_excess']:+.4f} floor={res['noise_floor']:+.4f} "
              f"-> excess={res['excess_above_floor']:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] {star}")
    return res


def run_corrected_pairs(panel, pairs, horizon_name="1m", n_noise_draws=10,
                        rng_seed=M.RNG_SEED, verbose=True):
    return [corrected_pair(panel, a, b, horizon_name, n_noise_draws, rng_seed,
                           verbose=verbose) for (a, b) in pairs]
