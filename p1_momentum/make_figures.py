"""Build every figure for the momentum paper (Momentum in the Nifty 500) from files in sources/.

Run from the paper folder:  python3 make_figures.py
Inputs (all downloaded read-only from the author's Drive; see REPO_NOTES.md):
  sources/decile_monthly_returns_2026-07-27.csv      Phase B baseline, 179 months x D1..D10
  sources/long_short_spread_2026-07-27.csv           Phase B baseline, D10-D1 monthly
  sources/liquid_universe_sweep_2026-08-05.csv       liquidity-universe test, turnover filter
  sources/liquid_universe_sweep_AMIHUD_2026-08-05.csv  same test, Amihud filter
Outputs: figures/*.pdf (grayscale-safe, no titles; captions live in main.tex).

The script also re-derives the headline baseline numbers from the archived CSVs and
asserts they match the values printed in the Phase B notebook, so a figure can never
silently drift from the numbers quoted in the text.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Load + reproduce the baseline headline numbers (guards against drift)
# ---------------------------------------------------------------------------
dec = pd.read_csv(SRC / "decile_monthly_returns_2026-07-27.csv", parse_dates=["rebalance_date"])
ls = pd.read_csv(SRC / "long_short_spread_2026-07-27.csv", parse_dates=["rebalance_date"])
assert len(dec) == 179 and len(ls) == 179, "expected 179 baseline months"

decile_mean_pct = dec[[f"D{i}" for i in range(1, 11)]].mean() * 100
expected = [0.2189, 0.8717, 0.9263, 1.1415, 1.0802, 1.1616, 1.3746, 1.5882, 1.5621, 1.9587]
assert np.allclose(decile_mean_pct.round(4).values, expected), decile_mean_pct
x = ls["spread"].to_numpy()
assert np.allclose((dec["D10"] - dec["D1"]).to_numpy(), x)
sharpe = x.mean() / x.std(ddof=1) * np.sqrt(12)
assert abs(sharpe - 0.8465) < 5e-4 and abs(x.mean() * 100 - 1.7398) < 5e-4

# Re-run the notebook's circular block bootstrap (12-month blocks, 10,000 draws,
# seed 20260724, Sharpe then mean drawn in the same loop) and confirm the CI.
rng = np.random.default_rng(20260724)
n, B = len(x), 12
nb = int(np.ceil(n / B))
xc = np.concatenate([x, x[:B]])
bs = np.empty(10_000)
for b in range(10_000):
    st = rng.integers(0, n, size=nb)
    s = np.concatenate([xc[i:i + B] for i in st])[:n]
    bs[b] = s.mean() / s.std(ddof=1) * np.sqrt(12)
lo, hi = np.percentile(bs, [2.5, 97.5])
assert abs(lo - 0.335) < 1e-3 and abs(hi - 1.503) < 1e-3, (lo, hi)
print(f"baseline reproduced: Sharpe {sharpe:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

# ---------------------------------------------------------------------------
# Figure 1: decile staircase (mean monthly forward return by momentum decile)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.6))
xs = np.arange(1, 11)
ax.bar(xs, decile_mean_pct.values, width=0.7, color="0.55", edgecolor="0.15", linewidth=0.5)
for xi, v in zip(xs, decile_mean_pct.values):
    ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="0.15")
ax.set_xticks(xs)
ax.set_xticklabels([f"D{i}" for i in xs])
ax.set_xlabel("12-1 momentum decile (D1 = lowest past return, D10 = highest)")
ax.set_ylabel("Mean return, % per month")
ax.set_ylim(0, 2.25)
ax.axhline(0, color="0.2", linewidth=0.6)
fig.savefig(OUT / "fig1_decile_staircase.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: liquidity-universe degradation curve, turnover vs Amihud, 95% CIs
# ---------------------------------------------------------------------------
turn = pd.read_csv(SRC / "liquid_universe_sweep_2026-08-05.csv")
amih = pd.read_csv(SRC / "liquid_universe_sweep_AMIHUD_2026-08-05.csv")
assert list(turn["floor"]) == list(amih["floor"])
labels = ["Top 30%", "Top 50%", "Top 70%", "Full"]
pos = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for df, off, mk, ls_, fill, name in [
        (turn, -0.07, "o", "-", "0.1", "Turnover filter (primary)"),
        (amih, +0.07, "s", "--", "white", "Amihud filter (robustness)")]:
    y = df["sharpe"].to_numpy()
    err = [y - df["sharpe_ci_low"].to_numpy(), df["sharpe_ci_high"].to_numpy() - y]
    ax.errorbar(pos + off, y, yerr=err, fmt=mk, linestyle=ls_, color="0.1", mfc=fill,
                mec="0.1", ms=5, lw=1.0, elinewidth=0.8, capsize=3, label=name)
ax.axhline(0, color="0.2", linewidth=0.6)
ax.axhline(0.30, color="0.45", linewidth=0.8, linestyle=":",
           label="Pre-registered screen (CI lower bound > 0.30)")
ax.set_xticks(pos)
ax.set_xticklabels([f"{l}\n({int(m)})" for l, m in zip(labels, turn["median_univ"])])
ax.set_xlabel("Liquidity universe (median names per month)")
ax.set_xlim(-0.45, 3.45)
ax.set_ylabel("D10$-$D1 annualised Sharpe (gross)")
ax.set_ylim(-0.2, 1.75)
ax.legend(loc="upper left", frameon=False, ncol=1)
fig.savefig(OUT / "fig2_liquidity_retention.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: cumulative D10-D1 spread (log scale), 2022+ shaded
# ---------------------------------------------------------------------------
cum = (1 + ls.set_index("rebalance_date")["spread"]).cumprod()
fig, ax = plt.subplots(figsize=(5.2, 2.5))
ax.axvspan(pd.Timestamp("2022-01-01"), cum.index.max() + pd.Timedelta(days=20),
           color="0.9", linewidth=0)
ax.text(pd.Timestamp("2022-03-01"), 1.15, "2022 onward\n(designated hold-out,\nused here)",
        fontsize=7, color="0.3", va="bottom")
ax.plot(cum.index, cum.values, color="0.1", linewidth=1.1)
ax.set_yscale("log")
ax.axhline(1, color="0.4", linewidth=0.6, linestyle="--")
ax.set_ylabel("Growth of 1 (log scale, gross)")
ax.set_xlabel("Rebalance month (return earned over the following month)")
fig.savefig(OUT / "fig3_cumulative_spread.pdf")
plt.close(fig)

# Retention ratios quoted in Table 2 (ratio of point estimates; no CI exists)
for name, df in [("turnover", turn), ("amihud", amih)]:
    full = df.loc[df["floor"].str.startswith("full"), "sharpe"].item()
    print(name, {f: round(s / full * 100, 1) for f, s in zip(df["floor"], df["sharpe"])})
print("figures written to", OUT)
