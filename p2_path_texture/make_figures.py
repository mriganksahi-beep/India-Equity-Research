"""Build every figure and every derived statistic for the path-texture paper (path texture).

Inputs (all in ./sources, downloaded read-only from the author's Drive):
  entropy_panel.parquet               M7 cached panel (pe4, pe3, sampen, dgw, mom, vol60, fwd)
  vr_panel.parquet                    Test D cached panel (vr, vr10, vr42, rv, mom, amihud, fwd)
  c_gate_monthly_correlations_*.csv   Test C per-month Spearman correlations
  c_gate_D1_shape_*.csv               Test C diagnostic 1 decile profile
  c_gate_D2_partial_*.csv             Test C diagnostic 2 raw vs partial correlations

Outputs:
  figures/fig1_pe_distribution.pdf
  figures/fig2_vr_distribution.pdf
  figures/fig3_testc_diagnostics.pdf
  derived_stats.csv   every number in the paper that was computed here (not read
                      from a findings file), with a label. claims.csv points here.

Everything labelled 'replication' re-runs the exact estimator logic of the source
notebooks on their cached panels. Everything labelled 'post hoc' or 'null' is new
to this paper and was NOT part of the pre-registered gates.

Run:  python3 make_figures.py      (deterministic; seeds fixed)
"""
from math import factorial, log
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
DEV_END = pd.Timestamp("2021-12-31")   # development period ends; hold-out = 2022-01-01 onward
MIN_NAMES = 100                        # same cross-section floor as the source notebooks

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.color": "0.6", "pdf.fonttype": 42,
})

stats = []  # (key, value, description)


def record(key, value, desc):
    stats.append((key, value, desc))


# ---------------------------------------------------------------------------
# Estimators copied verbatim (logic) from the source notebooks
# ---------------------------------------------------------------------------
def permutation_entropy(x, m=4, tau=1):
    """M7 notebook estimator: normalized Bandt-Pompe entropy, ties broken by argsort."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < m * tau + 1:
        return np.nan
    patterns = {}; count = 0
    for i in range(n - (m - 1) * tau):
        window = x[i:i + (m - 1) * tau + 1:tau]
        pat = tuple(np.argsort(window)); patterns[pat] = patterns.get(pat, 0) + 1; count += 1
    probs = np.array(list(patterns.values()), float) / count
    return (-np.sum(probs * np.log(probs))) / log(factorial(m))


def variance_ratio(daily_ret, q):
    """Test D notebook estimator: overlapping Lo-MacKinlay VR(q)."""
    r = np.asarray(daily_ret, float); r = r[~np.isnan(r)]; n = len(r)
    if n < q + 1:
        return np.nan
    mu = np.mean(r); sig1 = np.sum((r - mu) ** 2) / (n - 1)
    X = np.concatenate([[0.0], np.cumsum(r)])
    dq = (X[q:] - X[:-q]) - q * mu; mm = q * (n - q + 1) * (1.0 - q / n)
    return (np.sum(dq ** 2) / mm) / sig1


def zc(s):
    s = pd.Series(s, dtype=float); mu, sd = s.mean(), s.std(ddof=1)
    return (s - mu) / sd if sd > 0 else s * 0.0


def newey_west(series, lag=6):
    """Same Bartlett-kernel NW t-statistic (lag 6) as the source notebooks."""
    x = np.asarray(series, float); x = x[~np.isnan(x)]; T = len(x)
    mu = x.mean(); e = x - mu; var = (e @ e) / T
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1); var += 2 * w * (e[L:] @ e[:-L]) / T
    se = np.sqrt(var / T)
    return mu, se, mu / se, T


def circular_block_bootstrap_ci(x, block=6, reps=10000, seed=2026):
    """95% percentile CI for the mean of a monthly series (post hoc, this paper)."""
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); xx = np.concatenate([x, x[:block]])
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n, nb)
        means[b] = np.concatenate([xx[s:s + block] for s in starts])[:n].mean()
    return np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------------------
# 1. M7 step-1 gate: replication (full sample as run) and dev-only re-estimate
# ---------------------------------------------------------------------------
edf = pd.read_parquet(SRC / "entropy_panel.parquet")
record("m7_panel_rows", len(edf), "entropy panel stock-months (as cached by M7)")
record("m7_panel_first_rb", str(edf.rb.min().date()), "first rebalance in entropy panel")
record("m7_panel_last_rb", str(edf.rb.max().date()), "last rebalance in entropy panel")
record("m7_panel_months", edf.rb.nunique(), "rebalance months in entropy panel")


def m7_gate(df):
    unc, con = [], []
    for rb, g in df.groupby("rb"):
        d = g.dropna(subset=["pe4", "mom", "dgw", "vol60", "fwd"]).copy()
        if len(d) < MIN_NAMES:
            continue
        zpe, zmom, zdgw = zc(d.pe4), zc(d.mom), zc(d.dgw)
        Xu = np.column_stack([np.ones(len(d)), zpe, zmom])
        unc.append(np.linalg.lstsq(Xu, d.fwd.values, rcond=None)[0][1])
        Xc = np.column_stack([np.ones(len(d)), zpe, zmom, zdgw])
        con.append(np.linalg.lstsq(Xc, d.fwd.values, rcond=None)[0][1])
    return np.array(unc), np.array(con)


for tag, df in [("full", edf), ("dev", edf[edf.rb <= DEV_END])]:
    unc, con = m7_gate(df)
    for name, ser in [("unc", unc), ("dgw", con)]:
        mu, se, t, T = newey_west(ser)
        lo, hi = mu - 1.96 * se, mu + 1.96 * se
        blo, bhi = circular_block_bootstrap_ci(ser)
        record(f"m7_{tag}_{name}_coef", mu, f"M7 step-1 {name} mean monthly FM coef ({tag})")
        record(f"m7_{tag}_{name}_t", t, f"M7 step-1 {name} NW(6) t ({tag})")
        record(f"m7_{tag}_{name}_months", T, f"M7 step-1 months used ({tag})")
        record(f"m7_{tag}_{name}_nwci_lo", lo, "95% CI lower, coef +/- 1.96 NW SE (post hoc)")
        record(f"m7_{tag}_{name}_nwci_hi", hi, "95% CI upper, coef +/- 1.96 NW SE (post hoc)")
        record(f"m7_{tag}_{name}_cbb_lo", blo, "95% circular block bootstrap (block 6) lower (post hoc)")
        record(f"m7_{tag}_{name}_cbb_hi", bhi, "95% circular block bootstrap (block 6) upper (post hoc)")

# Stability diagnostic months as printed by the M7 notebook (copied from its output)
stab = {"2011-01-31": .452, "2012-04-30": .420, "2013-07-31": .472, "2014-10-31": .477,
        "2016-01-29": .427, "2017-04-28": .457, "2018-07-31": .438, "2019-10-31": .438,
        "2021-01-29": .416, "2022-04-29": .435, "2023-07-31": .432, "2024-10-31": .411}
dev_stab = [v for k, v in stab.items() if pd.Timestamp(k) <= DEV_END]
record("m7_stab_dev_mean", np.mean(dev_stab), "mean of the 9 dev-period stability months")
record("m7_stab_dev_n", len(dev_stab), "number of stability months inside 2010-2021")

dev = edf[edf.rb <= DEV_END]
record("pe_dev_rows", len(dev), "entropy-panel stock-months 2011-01..2021-12")
record("pe_dev_tickers", dev.ticker.nunique(), "tickers in dev slice")
for q in [0.01, 0.5, 0.99]:
    record(f"pe_dev_q{int(q*100):02d}", dev.pe4.quantile(q), f"PE(m=4) quantile {q}, dev")
record("pe_dev_min", dev.pe4.min(), "PE(m=4) minimum, dev")
record("pe_dev_share_ge095", (dev.pe4 >= 0.95).mean(), "share of dev stock-months with PE>=0.95")
record("pe_dev_xs_sd", dev.groupby("rb").pe4.std().mean(), "mean within-month cross-sectional SD of PE")
record("pe_dev_corr_pe3", dev[["pe4", "pe3"]].corr().iloc[0, 1], "corr PE4,PE3 dev (pooled)")
record("sampen_dev_median", dev.sampen.median(), "SampEn(m=2,r=0.2) median, dev")

# iid null for PE: ordinal patterns of any iid continuous series are uniform, so the
# null depends only on n and m. n = 232 = rows in [t-252, t-21] (no missing days).
rng = np.random.default_rng(7)
N_WIN = 232
null_pe = np.array([permutation_entropy(rng.standard_normal(N_WIN)) for _ in range(20000)])
record("pe_null_mean", null_pe.mean(), "iid null PE(m=4), n=232, 20000 draws (post hoc)")
record("pe_null_sd", null_pe.std(ddof=1), "iid null PE SD (post hoc)")
record("pe_null_q01", np.quantile(null_pe, .01), "iid null PE 1st pct (post hoc)")
record("pe_null_q99", np.quantile(null_pe, .99), "iid null PE 99th pct (post hoc)")
record("pe_noise_share", null_pe.var(ddof=1) / dev.groupby("rb").pe4.var().mean(),
       "iid-null PE variance / mean within-month cross-sectional PE variance (post hoc)")
record("pe_dev_share_inside_null99",
       ((dev.pe4 >= np.quantile(null_pe, .005)) & (dev.pe4 <= np.quantile(null_pe, .995))).mean(),
       "share of dev PE values inside central 99% of iid null (post hoc)")

# SampEn null (smaller, O(n^2) per draw)
def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, float); n = len(x); r = r * np.std(x, ddof=1)
    def phi(mm):
        T = np.array([x[i:i + mm] for i in range(n - mm + 1)]); C = 0
        for i in range(len(T)):
            C += np.sum(np.max(np.abs(T - T[i]), axis=1) <= r) - 1
        return C
    return -log(phi(m + 1) / phi(m))
null_se = np.array([sample_entropy(rng.standard_normal(N_WIN)) for _ in range(500)])
record("sampen_null_median", np.median(null_se), "iid null SampEn(2,0.2) median, 500 draws (post hoc)")

fig, ax = plt.subplots(figsize=(6.2, 2.7))
bins = np.linspace(0.95, 1.0, 76)
ax.hist(dev.pe4.clip(lower=0.95), bins=bins, density=True, color="0.35", alpha=0.85,
        label=f"NSE stock-months, 2011-2021 (n={len(dev):,})")
ax.hist(null_pe, bins=bins, density=True, histtype="step", color="black", lw=1.4,
        linestyle="--", label="iid null, same window length (simulated)")
ax.axvline(1.0, color="0.5", lw=0.8)
ax.set_xlabel("normalized permutation entropy, m = 4 (values < 0.95 shown at 0.95)")
ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig1_pe_distribution.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Test D gate 1: replication + post hoc q-robustness + null
# ---------------------------------------------------------------------------
vr = pd.read_parquet(SRC / "vr_panel.parquet")
record("d_panel_rows", len(vr), "VR panel stock-months")
record("d_panel_tickers", vr.ticker.nunique(), "VR panel tickers")
record("d_panel_first_rb", str(vr.rb.min().date()), "first VR rebalance")
record("d_panel_last_rb", str(vr.rb.max().date()), "last VR rebalance")
unc, res = [], []
for rb, g in vr.groupby("rb"):
    d = g.dropna(subset=["vr", "rv", "mom", "fwd"]).copy()
    if len(d) < MIN_NAMES:
        continue
    zvr, zrv, zmom = zc(d.vr), zc(np.log(d.rv.clip(lower=1e-9))), zc(d.mom)
    zam = zc(np.log(d.amihud.clip(lower=1e-12)))
    b = np.polyfit(zrv, zvr, 1); u = zvr - (b[0] * zrv + b[1])
    unc.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), zvr]), d.fwd.values, rcond=None)[0][1])
    Xc = np.column_stack([np.ones(len(d)), u, zrv, zmom, zam])
    res.append(np.linalg.lstsq(Xc, d.fwd.values, rcond=None)[0][1])
for name, ser in [("unc", unc), ("res", res)]:
    mu, se, t, T = newey_west(ser)
    blo, bhi = circular_block_bootstrap_ci(ser)
    record(f"d_{name}_coef", mu, f"D gate-1 {name} mean monthly coef (replication)")
    record(f"d_{name}_t", t, f"D gate-1 {name} NW(6) t (replication)")
    record(f"d_{name}_months", T, "months used")
    record(f"d_{name}_nwci_lo", mu - 1.96 * se, "95% CI lower, NW (post hoc)")
    record(f"d_{name}_nwci_hi", mu + 1.96 * se, "95% CI upper, NW (post hoc)")
    record(f"d_{name}_cbb_lo", blo, "95% circular block bootstrap lower (post hoc)")
    record(f"d_{name}_cbb_hi", bhi, "95% circular block bootstrap upper (post hoc)")
for c in ["vr10", "vr42"]:
    u2 = []
    for rb, g in vr.groupby("rb"):
        d = g.dropna(subset=[c, "rv", "mom", "fwd"])
        if len(d) < MIN_NAMES:
            continue
        u2.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), zc(d[c])]), d.fwd.values, rcond=None)[0][1])
    mu, se, t, T = newey_west(u2)
    record(f"d_{c}_unc_coef", mu, f"post hoc uncontrolled coef, {c}")
    record(f"d_{c}_unc_t", t, f"post hoc uncontrolled NW t, {c}")
record("vr_median", vr.vr.median(), "median VR(21), dev panel")
record("vr_xs_sd", vr.groupby("rb").vr.std().mean(), "mean within-month cross-sectional SD of VR(21)")
null_vr = np.array([variance_ratio(rng.standard_normal(N_WIN), 21) for _ in range(20000)])
record("vr_null_mean", null_vr.mean(), "iid Gaussian RW null VR(21), n=232 (post hoc)")
record("vr_null_sd", null_vr.std(ddof=1), "iid Gaussian RW null VR SD (post hoc)")
record("vr_null_median", np.median(null_vr), "iid Gaussian RW null VR(21) median (post hoc)")
record("vr_noise_share", null_vr.var(ddof=1) / vr.groupby("rb").vr.var().mean(),
       "null variance / mean cross-sectional variance (post hoc)")

fig, ax = plt.subplots(figsize=(6.2, 2.6))
bins = np.linspace(0, 3.5, 71)
ax.hist(vr.vr.clip(upper=3.5), bins=bins, density=True, color="0.35", alpha=0.85,
        label=f"NSE stock-months, 2011-2021 (n={len(vr):,})")
ax.hist(null_vr, bins=bins, density=True, histtype="step", color="black", lw=1.4,
        linestyle="--", label="iid Gaussian random walk (simulated)")
ax.axvline(1.0, color="0.5", lw=0.8)
ax.set_xlabel("variance ratio VR(21) over the formation window (values > 3.5 shown at 3.5)")
ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig2_vr_distribution.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Test C: CI on the gate statistic + diagnostic figure
# ---------------------------------------------------------------------------
cg = pd.read_csv(sorted(SRC.glob("c_gate_monthly_correlations_*.csv"))[0])
record("c_months", len(cg), "C gate months")
record("c_first_month", cg.month.iloc[0], "C first month")
record("c_last_month", cg.month.iloc[-1], "C last month")
for col in ["rho_full_amihud", "rho_liquid_amihud", "rho_full_turnover", "rho_liquid_turnover"]:
    s = cg[col].dropna().values
    record(f"c_{col}_mean", s.mean(), f"C mean {col}")
    lo, hi = circular_block_bootstrap_ci(s, block=6)
    record(f"c_{col}_cbb_lo", lo, "95% circular block bootstrap lower (post hoc)")
    record(f"c_{col}_cbb_hi", hi, "95% circular block bootstrap upper (post hoc)")
record("c_min_nfull_month", cg.loc[cg.n_full.idxmin(), "month"], "month with smallest full cross-section")
record("c_min_nfull", int(cg.n_full.min()), "smallest full cross-section")
d2 = pd.read_csv(sorted(SRC.glob("c_gate_D2_partial_*.csv"))[0])
record("c_d2_months", len(d2), "D2 months")
record("c_d2_raw_mean", d2.raw_corr.mean(), "D2 raw mean")
record("c_d2_partial_mean", d2.partial_corr.mean(), "D2 partial mean")
record("c_d2_collapse", 1 - d2.partial_corr.mean() / d2.raw_corr.mean(), "D2 collapse fraction")
d1 = pd.read_csv(sorted(SRC.glob("c_gate_D1_shape_*.csv"))[0])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
ax.plot(d1.rv_decile, d1.mean_amihud_rank, color="black", marker="o", ms=4, lw=1.4)
ax.set_xticks(range(1, 11)); ax.set_xlabel("RV60 decile, liquid core")
ax.set_ylabel("mean Amihud pct. rank"); ax.set_ylim(0.3, 0.7)
ax = axes[1]
t = pd.PeriodIndex(cg.month, freq="M").to_timestamp()
ax.plot(t, cg.rho_full_amihud, color="0.55", lw=1.0, label="full universe")
ax.plot(t, cg.rho_liquid_amihud, color="black", lw=1.4, label="liquid core")
ax.axhspan(0.20, 0.40, color="0.85", zorder=0, label="ambiguous band (0.20-0.40)")
ax.axhline(0, color="0.5", lw=0.6)
ax.set_ylabel("Spearman corr(RV60, Amihud)")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(FIG / "fig3_testc_diagnostics.pdf"); plt.close(fig)

out = pd.DataFrame(stats, columns=["key", "value", "description"])
out.to_csv(HERE / "derived_stats.csv", index=False)
print(out.to_string(index=False))
