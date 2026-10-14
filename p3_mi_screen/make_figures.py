"""Build every figure and derived table for Paper 3 from files in sources/.

Inputs (all produced by moonshot_mi_paper_rerun.ipynb, 2026-10-13, or by the
original 2026-08-10 runs):
  sources/landscape_v2_2026-10-13.csv      per-cell summaries, 95% MBB CIs
  sources/per_date_full_2026-10-13.parquet per-(feature,horizon,date) values
  sources/spearman_pairs_2026-10-13.csv    feature-pair Spearman + MI
  sources/mi_landscape_2026-08-10.csv      original screen (90% iid CIs)
Outputs: figures/*.pdf and derived/*.csv (numbers quoted in the paper).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC, FIG, DER = HERE / "sources", HERE / "figures", HERE / "derived"
FIG.mkdir(exist_ok=True); DER.mkdir(exist_ok=True)

FEATS = ["momentum_12_1", "reversal_1m", "rvol_60", "log_turnover_21",
         "overnight_tilt_21", "amihud_illiq"]
LABEL = {"momentum_12_1": "Momentum 12-1", "reversal_1m": "Reversal 1m",
         "rvol_60": "Realized vol 60d", "log_turnover_21": "Log turnover 21d",
         "overnight_tilt_21": "Overnight tilt 21d", "amihud_illiq": "Amihud (lagged)"}
HZ = ["1w", "1m", "3m", "6m", "1y"]
# Ordinal blue ramp (light -> dark = short -> long horizon); grayscale-safe.
HZ_COL = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#dcdbd5"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b0"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42})

land = pd.read_csv(SRC / "landscape_v2_2026-10-13.csv")
pdd = pd.read_parquet(SRC / "per_date_full_2026-10-13.parquet")
sp = pd.read_csv(SRC / "spearman_pairs_2026-10-13.csv")
old = pd.read_csv(SRC / "mi_landscape_2026-08-10.csv")


def implied_mi(rho, n):
    """Gaussian-copula MI implied by a Spearman rho, minus its null bias at n."""
    r = np.clip(2 * np.sin(np.pi * rho / 6), -0.999999, 0.999999)
    return -0.5 * np.log(1 - r * r) - 0.5 * (np.pi / 3) ** 2 / (n - 1)


# ---------------------------------------------------------------- derived
pdd["imp_dir"] = implied_mi(pdd.ic_dir, pdd.n)
pdd["imp_mag"] = implied_mi(pdd.ic_mag, pdd.n)
acc = (pdd.groupby(["feature", "horizon"], sort=False)
       .agg(mi_dir=("mi_dir", "mean"), imp_dir=("imp_dir", "mean"),
            imp_mag=("imp_mag", "mean"), ic_mean=("ic_dir", "mean"),
            ic_sd=("ic_dir", "std"), n_mean=("n", "mean"), n_dates=("n", "size"))
       .reset_index())
acc["ic_sd_null"] = 1 / np.sqrt(acc.n_mean - 1)
acc["dir_share"] = acc.imp_dir / acc.mi_dir
acc["mono_share"] = (acc.imp_dir + acc.imp_mag) / acc.mi_dir
acc["implied_from_mean_ic"] = implied_mi(acc.ic_mean, acc.n_mean)
acc.to_csv(DER / "mi_accounting.csv", index=False)

# CI width: moving-block 95% vs original iid 90%, and vs iid 95%
w = land.assign(w_mbb95=land.mi_dir_hi - land.mi_dir_lo,
                w_iid95=land.mi_dir_iid_hi - land.mi_dir_iid_lo,
                w_iid90_saved=land.saved_ci90_hi - land.saved_ci90_lo)
w["ratio_mbb_vs_iid95"] = w.w_mbb95 / w.w_iid95
w.groupby("horizon", sort=False)["ratio_mbb_vs_iid95"].agg(["mean", "min", "max"]) \
 .to_csv(DER / "ci_width_by_horizon.csv")

summary = {
    "cells": len(land),
    "mi_ci_excl_zero": int((land.mi_dir_lo > 0).sum()),
    "ic_ci_excl_zero": int(((land.ic_dir_lo > 0) | (land.ic_dir_hi < 0)).sum()),
    "icmag_ci_excl_zero": int(((land.ic_mag_lo > 0) | (land.ic_mag_hi < 0)).sum()),
    "class_direction": int((land["class"] == "direction-led").sum()),
    "class_magnitude": int((land["class"] == "magnitude-led").sum()),
    "class_mixed": int((land["class"] == "mixed/unclear").sum()),
    "class_nosignal": int((land["class"] == "no_signal").sum()),
    "dir_share_median": acc.dir_share.median(), "dir_share_min": acc.dir_share.min(),
    "dir_share_max": acc.dir_share.max(),
    "mono_share_median": acc.mono_share.median(),
    "mono_share_q25": acc.mono_share.quantile(.25), "mono_share_q75": acc.mono_share.quantile(.75),
    "ic_sd_min": acc.ic_sd.min(), "ic_sd_max": acc.ic_sd.max(),
    "ic_sd_null_mean": acc.ic_sd_null.mean(),
    "n_dates_min": int(land.n_dates.min()), "n_dates_max": int(land.n_dates.max()),
    "names_min": land.mean_n.min(), "names_max": land.mean_n.max(),
}
pd.Series(summary).to_csv(DER / "headline_numbers.csv", header=["value"])

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
ypos = {f: i for i, f in enumerate(FEATS[::-1])}
off = np.linspace(-0.3, 0.3, len(HZ))
for k, (h, col) in enumerate(zip(HZ, HZ_COL)):
    s = land[land.horizon == h].set_index("feature").loc[FEATS]
    y = np.array([ypos[f] for f in FEATS]) + off[k]
    for ax, (m, lo, hi) in zip(axes, [("mi_dir", "mi_dir_lo", "mi_dir_hi"),
                                      ("ic_dir", "ic_dir_lo", "ic_dir_hi")]):
        ax.hlines(y, s[lo], s[hi], color=col, lw=1.4)
        ax.plot(s[m], y, "o", ms=3.6, color=col, mec="white", mew=0.6,
                label=h if ax is axes[0] else None)
axes[0].set_yticks(range(len(FEATS))); axes[0].set_yticklabels([LABEL[f] for f in FEATS[::-1]])
axes[0].set_xlabel(r"MI$_{\mathrm{excess}}$ (nats), 95% block CI"); axes[0].set_xlim(left=0)
axes[1].axvline(0, color=MUTED, lw=0.8)
axes[1].set_xlabel("Mean rank IC, 95% block CI")
for ax in axes: ax.grid(axis="y", visible=False)
axes[0].legend(title="Horizon", ncol=5, loc="lower center", bbox_to_anchor=(1.0, 1.0),
               handletextpad=0.2, columnspacing=0.9, fontsize=7.5, title_fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig1_landscape.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True, sharex=True)
for ax, h in zip(axes, ["1m", "1y"]):
    a = acc[acc.horizon == h].set_index("feature").loc[FEATS[::-1]]
    d = a.imp_dir.clip(lower=0); m = a.imp_mag.clip(lower=0)
    rest = (a.mi_dir - d - m).clip(lower=0)
    y = np.arange(len(a))
    ax.barh(y, d, color=BLUE, height=0.62, label="Per-date monotone direction (from rank IC)")
    ax.barh(y, m, left=d, color=ORANGE, height=0.62, edgecolor="white", linewidth=1,
            label="Monotone extremeness (rank IC on |rank$-$0.5|)")
    ax.barh(y, rest, left=d + m, color=GRAY, height=0.62, edgecolor="white", linewidth=1,
            label="Remainder")
    over = (d + m) > a.mi_dir
    ax.plot(a.mi_dir, y, "|", color=INK, ms=11, mew=1.3)
    ax.set_xlabel(f"Nats, {h} horizon"); ax.grid(axis="y", visible=False)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[f] for f in a.index])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=7, handlelength=1.2,
           columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(FIG / "fig2_accounting.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.0), sharex=True)
for ax, f in zip(axes, ["log_turnover_21", "momentum_12_1"]):
    s = pdd[(pdd.feature == f) & (pdd.horizon == "1m")].sort_values("date")
    ax.bar(s.date, s.ic_dir, width=22, color=np.where(s.ic_dir >= 0, BLUE, ORANGE))
    band = 1.96 / np.sqrt(s.n.mean() - 1)
    ax.axhspan(-band, band, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    mi = land[(land.feature == f) & (land.horizon == "1m")].iloc[0]
    ax.text(0.01, 0.95, f"{LABEL[f]}, 1m:  mean IC {mi.ic_dir:+.3f},  "
            f"MI$_{{\\mathrm{{excess}}}}$ {mi.mi_dir:.4f}", transform=ax.transAxes,
            va="top", fontsize=7.5, color=INK)
    ax.set_ylabel("Rank IC"); ax.grid(axis="x", visible=False); ax.set_ylim(-0.6, 0.6)
fig.tight_layout(); fig.savefig(FIG / "fig3_ic_series.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(3.4, 3.0))
lo = 0.003
ax.plot([lo, 2], [lo, 2], color=MUTED, lw=0.8, ls="--")
ax.scatter(sp.copula_implied_mi_excess.clip(lower=lo), sp.ksg_mi_excess, s=22,
           color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
short = {"momentum_12_1": "Mom", "reversal_1m": "Rev", "rvol_60": "Vol",
         "log_turnover_21": "Turn", "overnight_tilt_21": "Ovn", "amihud_illiq": "Ami"}
for _, r in sp.iterrows():
    if r.ksg_mi_excess - r.copula_implied_mi_excess > 0.03 or r.ksg_mi_excess > 1:
        ax.annotate(f"{short[r.feature_a]}–{short[r.feature_b]}",
                    (max(r.copula_implied_mi_excess, lo), r.ksg_mi_excess),
                    xytext=(4, -3), textcoords="offset points", fontsize=7, color=INK)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("MI implied by per-date Spearman (nats)")
ax.set_ylabel("KSG MI$_{\\mathrm{excess}}$ (nats)")
fig.tight_layout(); fig.savefig(FIG / "fig4_redundancy.pdf"); plt.close(fig)

print(pd.Series(summary).round(4).to_string())
print(acc[["feature", "horizon", "mi_dir", "imp_dir", "imp_mag", "dir_share",
           "mono_share", "ic_sd"]].round(4).to_string(index=False))
print(pd.read_csv(DER / "ci_width_by_horizon.csv").round(3).to_string(index=False))
