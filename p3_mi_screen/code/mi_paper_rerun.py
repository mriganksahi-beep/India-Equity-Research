"""
mi_paper_rerun.py
=================
Reproduction + robustness pass for the MI-screen working paper.

Read-only characterization on the 2010-2021 DEV period. Nothing here selects a
feature or touches the 2022+ hold-out: every cross-section is built by
mi_screen's own guards (flag == 1, no fill, forward window <= TRAIN_END), and
an explicit audit re-checks the hold-out wall on every date used.

What this adds on top of mi_screen.py / conditional_mi.py
---------------------------------------------------------
1. REPRODUCTION CAPTURE. Re-runs M.run_screen exactly as the 2026-08-10 driver
   did, while recording the per-date MI_excess values it computes (the original
   run discarded them). The recorder wraps M.per_date_mi_excess without
   touching the RNG stream, so the reproduced numbers are the original
   procedure, not an approximation of it.

2. HONEST CIs. The original CIs were (a) 90% (5th/95th percentiles) and
   (b) an iid bootstrap over months, which is too narrow when forward windows
   overlap (3m/6m/1y on a monthly grid). Here: 95% circular moving-block
   bootstrap with block length >= the horizon in months.

3. DIRECTION vs MAGNITUDE (pre-registered, see MAG_SHARE_THRESHOLD).
   MI is sign-blind. For each (feature, horizon) cell we also compute
     * rank IC  = Spearman(feature, forward return)          -> direction
     * MI_mag   = MI(feature ; |fwd_rank - 0.5|)             -> size of move
     * IC_mag   = Spearman(feature, |fwd_rank - 0.5|)
   By the data-processing inequality MI_mag <= MI_dir in population, so
   share = MI_mag / MI_dir is (up to estimation noise) in [0, 1]: the fraction
   of a feature's MI that is about how FAR a stock moves rather than WHERE.

4. REDUNDANCY CROSS-CHECK. Per-date Spearman between every feature pair, plus
   the MI a Gaussian copula with that Spearman would imply. If KSG MI ~ the
   copula-implied MI, the dependence is essentially monotone and rank
   correlation already sees it; if KSG MI >> implied, there is genuinely
   non-monotone structure. This tests (rather than asserts) the claim that MI
   "catches what rank correlation misses".

5. CONDITIONAL-MI CAPTURE. Re-runs C.run_corrected_pairs as the 2026-09
   correction notebook did, recording the per-date (A|B excess - noise floor)
   series so it too gets a 95% moving-block CI.

NOTE ON EFFECTIVE SETTINGS (found while reviewing the code): the original
drivers set M.N_SHUFFLE = 15 and M.N_BOOT = 500 AFTER import, but
per_date_mi_excess(n_shuffle=N_SHUFFLE) and summarise(n_boot=N_BOOT) bind
their defaults at definition time. The screen therefore actually ran with
40 shuffles and 300 bootstrap draws. effective_settings() reports what
really ran; the paper must quote those, not the printed values.
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd

import mi_screen as M

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CI_LEVEL = 0.95
N_BOOT_MBB = 2000
# Block length in MONTHS (= rebalance steps). >= horizon so overlapping forward
# windows sit inside one block; floor of 3 to absorb slow feature persistence.
BLOCK_MONTHS = {"1w": 3, "1m": 3, "3m": 3, "6m": 6, "1y": 12}
SEED_EXT = M.RNG_SEED + 1          # separate stream: never perturbs reproduction

# Pre-registered classification rule (fixed BEFORE the real-data run).
MAG_SHARE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# SETTINGS AUDIT
# ---------------------------------------------------------------------------
def effective_settings() -> dict:
    """What the screen ACTUALLY uses, read from the bound defaults."""
    pde_default = inspect.signature(M.per_date_mi_excess).parameters["n_shuffle"].default
    summ_default = inspect.signature(M.summarise).parameters["n_boot"].default
    return {
        "per_date_mi_excess.n_shuffle(default bound at import)": pde_default,
        "summarise.n_boot(default bound at import)": summ_default,
        "module N_SHUFFLE attr (may differ!)": M.N_SHUFFLE,
        "module N_BOOT attr (may differ!)": M.N_BOOT,
        "KNN_K": M.KNN_K,
        "original CI percentiles": (M.CI_LOW, M.CI_HIGH),
        "TRAIN_START": str(M.TRAIN_START.date()),
        "TRAIN_END (hold-out wall)": str(M.TRAIN_END.date()),
        "MIN_STOCKS_PER_DATE": M.MIN_STOCKS_PER_DATE,
        "RNG_SEED": M.RNG_SEED,
        "SEED_EXT": SEED_EXT,
        "CI_LEVEL (new)": CI_LEVEL,
        "N_BOOT_MBB (new)": N_BOOT_MBB,
        "BLOCK_MONTHS (new)": BLOCK_MONTHS,
        "MAG_SHARE_THRESHOLD (pre-registered)": MAG_SHARE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# MOVING-BLOCK BOOTSTRAP
# ---------------------------------------------------------------------------
def mbb_means(X: np.ndarray, block: int, n_boot: int,
              rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap of column means.

    X: (m, k) array, rows in DATE ORDER (the bootstrap relies on adjacency).
    All k columns are resampled with the SAME block indices, so paired
    statistics (e.g. a ratio of two columns' means) get a coherent CI.
    block=1 reduces to the ordinary iid bootstrap. Returns (n_boot, k).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    m = X.shape[0]
    if m < 3:
        raise ValueError(f"mbb_means needs >= 3 rows, got {m}")
    if np.isnan(X).any():
        raise ValueError("mbb_means received NaN; drop/handle explicitly upstream")
    L = int(min(max(block, 1), m))
    n_blocks = int(np.ceil(m / L))
    offsets = np.arange(L)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, m, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % m).ravel()[:m]
        out[b] = X[idx].mean(axis=0)
    return out


def ci_from_draws(draws: np.ndarray, level: float = CI_LEVEL):
    a = (1.0 - level) / 2.0
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1.0 - a))


# ---------------------------------------------------------------------------
# PER-DATE STATISTICS ON A LIST OF DateCrossSection
# ---------------------------------------------------------------------------
def per_date_rank_ic(css) -> np.ndarray:
    """Spearman per date. Inputs are already average-tie ranks in (0,1), so the
    Pearson correlation of the ranks IS the Spearman correlation."""
    return np.array([np.corrcoef(c.feat_rank, c.fwd_rank)[0, 1] for c in css])


def magnitude_cross_sections(css):
    """Same stocks, same dates; target replaced by the rank of |fwd_rank - 0.5|,
    i.e. how far each stock moved from the cross-sectional median, ignoring
    direction. Rank pairs i and n+1-i tie exactly; average-tie ranking keeps the
    tie and the KSG estimator's internal jitter breaks it (validated in the
    self-test below)."""
    out = []
    for c in css:
        mag = M._rank01(np.abs(c.fwd_rank - 0.5))
        out.append(M.DateCrossSection(date=c.date, feat_rank=c.feat_rank,
                                      fwd_rank=mag, n=c.n))
    return out


# ---------------------------------------------------------------------------
# 1. REPRODUCTION WITH CAPTURE
# ---------------------------------------------------------------------------
class _Recorder:
    """Wraps a per-date function; passes every argument through unchanged
    (so bound defaults and the RNG stream are exactly the original's) and
    stores (dates, n, output) for each call."""
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, items, rng, *args, **kwargs):
        out = self.fn(items, rng, *args, **kwargs)
        self.calls.append((list(items), np.array(out, copy=True)))
        return out


def reproduce_screen(panel, verbose=True):
    """Run M.run_screen exactly as the original driver did and capture the
    per-date MI_excess series. Returns (results, per_date_long_df)."""
    rec = _Recorder(M.per_date_mi_excess)
    original = M.per_date_mi_excess
    M.per_date_mi_excess = rec
    try:
        results = M.run_screen(panel, verbose=verbose)
    finally:
        M.per_date_mi_excess = original

    # run_screen iterates FEATURES then HORIZONS, calling per_date_mi_excess
    # once per cell that has >= 3 dates. Map calls back to cells in that order.
    cells = [(f, h) for f in M.FEATURES for h in M.HORIZONS
             if "mi_excess" in results.get(f, {}).get(h, {})]
    if len(cells) != len(rec.calls):
        raise RuntimeError(f"capture mismatch: {len(cells)} cells vs {len(rec.calls)} calls")

    rows = []
    for (f, h), (css, vals) in zip(cells, rec.calls):
        # Consistency: captured mean must equal the reported point estimate.
        rep = results[f][h]["mi_excess"]
        if not np.isclose(vals.mean(), rep, rtol=0, atol=1e-12):
            raise RuntimeError(f"captured mean != reported for {f}/{h}")
        for c, v in zip(css, vals):
            rows.append(dict(feature=f, horizon=h, date=c.date, n=c.n, mi_dir=float(v)))
    return results, pd.DataFrame(rows)


def compare_to_saved(results, saved_landscape: pd.DataFrame, tol=2e-3) -> pd.DataFrame:
    """Side-by-side of reproduced vs saved (2026-08-10) landscape."""
    rows = []
    for _, r in saved_landscape.iterrows():
        cur = results.get(r["feature"], {}).get(r["horizon"], {})
        rows.append(dict(feature=r["feature"], horizon=r["horizon"],
                         saved=r["mi_excess"], reproduced=cur.get("mi_excess", np.nan),
                         saved_n=r["n_dates"], reproduced_n=cur.get("n_dates", np.nan)))
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df.saved - df.reproduced).abs()
    df["match"] = (df.abs_diff <= tol) & (df.saved_n == df.reproduced_n)
    return df


# ---------------------------------------------------------------------------
# HOLD-OUT AUDIT
# ---------------------------------------------------------------------------
def audit_holdout(panel, per_date: pd.DataFrame) -> int:
    """Re-verify, independently of the guard inside cross_section, that no
    forward window used reaches past TRAIN_END. Returns rows checked; raises
    on any violation."""
    idx = panel.dates
    n_checked = 0
    for (f, h), g in per_date.groupby(["feature", "horizon"]):
        hd = M.HORIZONS[h]
        pos = idx.get_indexer(pd.DatetimeIndex(g["date"]))
        if (pos < 0).any():
            raise RuntimeError(f"{f}/{h}: date not in panel index")
        fwd_end = idx[pos + hd]
        if (fwd_end > M.TRAIN_END).any():
            bad = g["date"][fwd_end > M.TRAIN_END].iloc[0]
            raise RuntimeError(f"HOLD-OUT LEAK {f}/{h}: rebalance {bad} reads past TRAIN_END")
        n_checked += len(g)
    return n_checked


# ---------------------------------------------------------------------------
# 2. DIRECTION (IC) + MAGNITUDE PASS
# ---------------------------------------------------------------------------
def direction_magnitude_pass(panel, per_date_dir: pd.DataFrame,
                             n_shuffle: int, verbose=True) -> pd.DataFrame:
    """Rebuild each cell's cross-sections deterministically (no RNG involved),
    confirm they are the SAME dates as the reproduction, then add rank IC,
    magnitude MI_excess (same estimator, same shuffle count) and magnitude IC."""
    rng = np.random.default_rng(SEED_EXT)
    rebal = M.month_end_rebalance_dates(panel)
    fwd_frames = {h: M.forward_return(panel, hd) for h, hd in M.HORIZONS.items()}
    out = []
    for f, fn in M.FEATURES.items():
        fw = fn(panel)
        for h, hd in M.HORIZONS.items():
            ref = per_date_dir[(per_date_dir.feature == f) & (per_date_dir.horizon == h)]
            if ref.empty:
                continue
            css, _ = M._collect_cross_sections(panel, fw, fwd_frames[h], rebal, hd)
            got = pd.DatetimeIndex([c.date for c in css])
            if not got.equals(pd.DatetimeIndex(ref["date"])):
                raise RuntimeError(f"{f}/{h}: rebuilt cross-sections differ from reproduction")
            ic = per_date_rank_ic(css)
            css_mag = magnitude_cross_sections(css)
            mi_mag = M.per_date_mi_excess(css_mag, rng, n_shuffle=n_shuffle)
            ic_mag = per_date_rank_ic(css_mag)
            g = ref.copy()
            g["ic_dir"] = ic
            g["mi_mag"] = mi_mag
            g["ic_mag"] = ic_mag
            out.append(g)
            if verbose:
                print(f"  {f:>18} | {h:>3}: IC={ic.mean():+.4f}  "
                      f"MI_dir={g.mi_dir.mean():+.4f}  MI_mag={mi_mag.mean():+.4f}  "
                      f"IC_mag={ic_mag.mean():+.4f}  (n_dates={len(g)})")
    return pd.concat(out, ignore_index=True)


def classify(row) -> str:
    """Pre-registered rule (MAG_SHARE_THRESHOLD fixed before the run)."""
    if not row["mi_dir_lo"] > 0:
        return "no_signal"
    if row["mag_share"] >= MAG_SHARE_THRESHOLD and row["mi_mag_lo"] > 0:
        return "magnitude-led"
    if row["mag_share"] < MAG_SHARE_THRESHOLD and (row["ic_dir_lo"] > 0 or row["ic_dir_hi"] < 0):
        return "direction-led"
    return "mixed/unclear"


def summarise_cells(per_date: pd.DataFrame, saved_landscape: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (feature, horizon): point estimates + 95% MBB CIs for every
    per-date statistic, the magnitude share with a paired CI, the iid-95% CI for
    comparison, and the pre-registered classification."""
    rng = np.random.default_rng(SEED_EXT + 7)
    stats = ["mi_dir", "ic_dir", "mi_mag", "ic_mag"]
    rows = []
    for (f, h), g in per_date.groupby(["feature", "horizon"], sort=False):
        g = g.sort_values("date")
        X = g[stats].to_numpy(float)
        L = BLOCK_MONTHS[h]
        d_mbb = mbb_means(X, L, N_BOOT_MBB, rng)
        d_iid = mbb_means(X[:, :1], 1, N_BOOT_MBB, rng)
        r = dict(feature=f, horizon=h, n_dates=len(g), mean_n=float(g.n.mean()), block=L)
        for k, s in enumerate(stats):
            r[s] = float(X[:, k].mean())
            r[f"{s}_lo"], r[f"{s}_hi"] = ci_from_draws(d_mbb[:, k])
        r["mi_dir_iid_lo"], r["mi_dir_iid_hi"] = ci_from_draws(d_iid[:, 0])
        share = d_mbb[:, 2] / d_mbb[:, 0]
        r["mag_share"] = r["mi_mag"] / r["mi_dir"] if r["mi_dir"] > 0 else np.nan
        r["mag_share_lo"], r["mag_share_hi"] = ci_from_draws(share[np.isfinite(share)])
        r["ic_tstat_naive"] = float(X[:, 1].mean() / (X[:, 1].std(ddof=1) / np.sqrt(len(X))))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["class"] = df.apply(classify, axis=1)
    if saved_landscape is not None:
        s = saved_landscape.set_index(["feature", "horizon"])
        df["saved_ci90_lo"] = [s.loc[(a, b), "ci_low"] for a, b in zip(df.feature, df.horizon)]
        df["saved_ci90_hi"] = [s.loc[(a, b), "ci_high"] for a, b in zip(df.feature, df.horizon)]
    return df


# ---------------------------------------------------------------------------
# 4. REDUNDANCY CROSS-CHECK (Spearman + copula-implied MI)
# ---------------------------------------------------------------------------
def gaussian_copula_mi_from_spearman(rho_s: float) -> float:
    """MI (nats) of a Gaussian copula with Spearman rho_s:
    Pearson r = 2 sin(pi * rho_s / 6);  MI = -0.5 * ln(1 - r^2)."""
    r = 2.0 * np.sin(np.pi * rho_s / 6.0)
    r = np.clip(r, -0.999999, 0.999999)
    return float(-0.5 * np.log(1.0 - r * r))


def spearman_matrix(panel, features: dict | None = None) -> pd.DataFrame:
    """Per-date Spearman for every feature pair on the same eligibility as the
    redundancy matrix (flag == 1, both present, >= MIN_STOCKS_PER_DATE).
    Long output: pair, mean rho, 95% MBB CI, copula-implied MI."""
    features = features or M.FEATURES
    rng = np.random.default_rng(SEED_EXT + 11)
    rebal = M.month_end_rebalance_dates(panel)
    frames = {k: fn(panel) for k, fn in features.items()}
    names = list(features)
    rows = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            fa, fb = frames[names[a]], frames[names[b]]
            rhos, mis = [], []
            for t in rebal:
                elig = (panel.flag.loc[t] == 1) & fa.loc[t].notna() & fb.loc[t].notna()
                if int(elig.sum()) < M.MIN_STOCKS_PER_DATE:
                    continue
                ra = M._rank01(fa.loc[t][elig].to_numpy(float))
                rb = M._rank01(fb.loc[t][elig].to_numpy(float))
                n = int(elig.sum())
                rho = float(np.corrcoef(ra, rb)[0, 1])
                rhos.append(rho)
                # Null-bias correction so this is comparable to KSG MI_EXCESS:
                # under independence Var(rho_s) ~ 1/(n-1) and r ~ (pi/3) rho_s,
                # so E[-0.5 ln(1-r^2)] ~ 0.5 (pi/3)^2 / (n-1). Subtract it.
                null_bias = 0.5 * (np.pi / 3.0) ** 2 / (n - 1)
                mis.append(gaussian_copula_mi_from_spearman(rho) - null_bias)
            if len(rhos) < 3:
                continue
            lo, hi = ci_from_draws(mbb_means(np.array(rhos), 3, N_BOOT_MBB, rng)[:, 0])
            rows.append(dict(feature_a=names[a], feature_b=names[b], n_dates=len(rhos),
                             spearman=float(np.mean(rhos)), spearman_lo=lo, spearman_hi=hi,
                             copula_implied_mi_excess=float(np.mean(mis))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. CONDITIONAL-MI REPRODUCTION WITH CAPTURE
# ---------------------------------------------------------------------------
def reproduce_corrected_cmi(panel, C, pairs, horizon_name="1m", n_noise_draws=10,
                            verbose=True):
    """Re-run C.run_corrected_pairs, capturing the per-date (A|B excess - floor)
    series. Each corrected_pair makes 1 call for A|B then n_noise_draws floor
    calls to C._npeet_cmi_excess_per_date, in that order."""
    rec = _Recorder(C._npeet_cmi_excess_per_date)
    original = C._npeet_cmi_excess_per_date
    C._npeet_cmi_excess_per_date = rec
    try:
        results = C.run_corrected_pairs(panel, pairs, horizon_name=horizon_name,
                                        n_noise_draws=n_noise_draws, verbose=verbose)
    finally:
        C._npeet_cmi_excess_per_date = original

    per_call = 1 + n_noise_draws
    if len(rec.calls) != per_call * len(pairs):
        raise RuntimeError(f"CMI capture mismatch: {len(rec.calls)} calls for {len(pairs)} pairs")

    h = M.HORIZONS[horizon_name]
    fwd = M.forward_return(panel, h)
    rebal = M.month_end_rebalance_dates(panel)
    rng = np.random.default_rng(SEED_EXT + 13)
    long_rows, summ = [], []
    for p, ((a_name, b_name), res) in enumerate(zip(pairs, results)):
        block = rec.calls[p * per_call:(p + 1) * per_call]
        pde_B = block[0][1]
        floor = np.mean([c[1] for c in block[1:]], axis=0)
        diff = pde_B - floor
        if not np.isclose(diff.mean(), res["excess_above_floor"], rtol=0, atol=1e-12):
            raise RuntimeError(f"{a_name}|{b_name}: captured excess != reported")
        fa = M.FEATURES[a_name](panel); fb = M.FEATURES[b_name](panel)
        dates = [t for t in rebal if C._eligible_triple(panel, fa, fb, fwd, t, h) is not None]
        if len(dates) != len(diff):
            raise RuntimeError(f"{a_name}|{b_name}: date count mismatch")
        lo, hi = ci_from_draws(mbb_means(diff, BLOCK_MONTHS[horizon_name], N_BOOT_MBB, rng)[:, 0])
        r = dict(res)
        r.update(ci95_mbb_lo=lo, ci95_mbb_hi=hi,
                 verdict=("additive" if lo > 0 else "redundant" if hi < 0 else "inconclusive"))
        summ.append(r)
        for t, v, vb, vf in zip(dates, diff, pde_B, floor):
            long_rows.append(dict(pair=f"{a_name}|{b_name}", date=t, cond_excess=float(vb),
                                  floor=float(vf), excess_above_floor=float(v)))
    return pd.DataFrame(summ), pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# ESTIMATOR CALIBRATION (synthetic, known answer)
# ---------------------------------------------------------------------------
def ksg_calibration(rhos=(0.1, 0.2, 0.3, 0.6, 0.9, 0.98), n=480, reps=30,
                    n_shuffle=10, seed=3) -> pd.DataFrame:
    """KSG MI_excess (the screen's exact estimator + shuffle null, on ranks)
    versus the TRUE MI of a bivariate Gaussian with correlation rho, at a
    cross-section size like the real panel (~480 names). Tells the reader how
    far a reported MI_excess can be read as a true MI, and validates comparing
    KSG MI against copula-implied MI in the redundancy cross-check."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho in rhos:
        ex = []
        for _ in range(reps):
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
            a, b = M._rank01(z[:, 0]), M._rank01(z[:, 1])
            raw = M._ksg_mi(a, b, rng)
            null = np.mean([M._ksg_mi(a, b[rng.permutation(n)], rng) for _ in range(n_shuffle)])
            ex.append(raw - null)
        true = -0.5 * np.log(1 - rho ** 2)
        rows.append(dict(rho=rho, n=n, reps=reps, true_mi=true,
                         ksg_excess_mean=float(np.mean(ex)),
                         ksg_excess_sd=float(np.std(ex, ddof=1)),
                         ratio=float(np.mean(ex) / true)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SELF-TEST FOR THE NEW MACHINERY (known answers; no real data)
# ---------------------------------------------------------------------------
def selftest(n_dates=120, n_stocks=400, n_shuffle=10, seed=5, verbose=True):
    """Gate: must pass before the real run.

    (a) MBB: on an iid series the MBB CI ~ iid CI width; on a persistent AR(1)
        series the MBB CI is clearly wider than iid.
    (b) Direction feature (y = 0.3x + e): IC significant, magnitude share < 0.5
        -> 'direction-led'.
    (c) Volatility feature (y = e * exp(0.8x)): IC ~ 0, magnitude MI significant,
        share >= 0.5 -> 'magnitude-led'.
    (d) Noise feature: 'no_signal'.
    (e) Copula identity: Spearman of a Gaussian pair recovers the true MI.
    """
    rng = np.random.default_rng(seed)

    # (a) MBB behaviour
    iid = rng.standard_normal(144)
    ar = np.empty(144); ar[0] = rng.standard_normal()
    for i in range(1, 144):
        ar[i] = 0.85 * ar[i - 1] + rng.standard_normal()
    w = lambda x, L: np.subtract(*ci_from_draws(mbb_means(x, L, 3000, rng)[:, 0])[::-1])
    r_iid = w(iid, 12) / w(iid, 1)
    r_ar = w(ar, 12) / w(ar, 1)
    if verbose:
        print(f"[mbb] width ratio MBB/iid: iid series={r_iid:.2f} (expect ~1), "
              f"AR(0.85) series={r_ar:.2f} (expect >1.5)")
    assert 0.7 < r_iid < 1.4, "MBB widens an iid series -> bug"
    assert r_ar > 1.5, "MBB fails to widen a persistent series -> bug"

    # (b)-(d) synthetic cross-sections
    dates = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    def make(kind):
        css = []
        for t in dates:
            x = rng.standard_normal(n_stocks)
            e = rng.standard_normal(n_stocks)
            if kind == "direction":
                y = 0.3 * x + e
            elif kind == "vol":
                y = e * np.exp(0.8 * x)
            else:
                y = e
            css.append(M.DateCrossSection(date=t, feat_rank=M._rank01(x),
                                          fwd_rank=M._rank01(y), n=n_stocks))
        return css
    rows = []
    for kind in ["direction", "vol", "noise"]:
        css = make(kind)
        mi_dir = M.per_date_mi_excess(css, rng, n_shuffle=n_shuffle)
        cm = magnitude_cross_sections(css)
        rows += [dict(feature=kind, horizon="1m", date=c.date, n=c.n, mi_dir=a,
                      ic_dir=b, mi_mag=m, ic_mag=d)
                 for c, a, b, m, d in zip(css, mi_dir, per_date_rank_ic(css),
                                          M.per_date_mi_excess(cm, rng, n_shuffle=n_shuffle),
                                          per_date_rank_ic(cm))]
    tab = summarise_cells(pd.DataFrame(rows)).set_index("feature")
    if verbose:
        print(tab[["mi_dir", "mi_dir_lo", "ic_dir", "ic_dir_lo", "ic_dir_hi",
                   "mi_mag", "mi_mag_lo", "mag_share", "class"]].round(4).to_string())
    assert tab.loc["direction", "class"] == "direction-led", "direction plant misclassified"
    assert tab.loc["vol", "class"] == "magnitude-led", "volatility plant misclassified"
    assert tab.loc["noise", "class"] == "no_signal", "noise flagged as signal"
    assert tab.loc["vol", "ic_dir_lo"] < 0 < tab.loc["vol", "ic_dir_hi"], "vol plant shows spurious IC"

    # (e) copula identity
    rho = 0.6
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=20000)
    rs = np.corrcoef(M._rank01(z[:, 0]), M._rank01(z[:, 1]))[0, 1]
    true_mi = -0.5 * np.log(1 - rho ** 2)
    est = gaussian_copula_mi_from_spearman(rs)
    if verbose:
        print(f"[copula] true MI={true_mi:.4f}  implied from Spearman={est:.4f}")
    assert abs(est - true_mi) < 0.01, "copula-implied MI formula wrong"

    # (f) estimator calibration at dependence levels where comparisons are made
    cal = ksg_calibration(rhos=(0.6, 0.9), reps=15, seed=seed)
    if verbose:
        print(cal[["rho", "true_mi", "ksg_excess_mean", "ratio"]].round(4).to_string(index=False))
    assert cal.ratio.between(0.8, 1.2).all(), "KSG badly mis-calibrated at moderate/strong dependence"

    if verbose:
        print("\nALL RE-RUN SELF-TEST ASSERTIONS PASSED.")
    return tab
