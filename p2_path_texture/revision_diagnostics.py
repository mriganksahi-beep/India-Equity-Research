"""Round-2 revision diagnostics for the path-texture paper (all POST HOC).

None of these analyses were part of the original gate chains. They were added
during revision (2026-10-14) in response to the external review. Outputs:
  revision_stats.csv           every number quoted from this script
  revision_tie_detail.csv      per stock-month tie diagnostics (35-ticker subset)
  figures/fig4_ties.pdf        PE under alternative tie rules, low-tail vs random tickers

Inputs (read-only copies of author's Drive files):
  sources/entropy_panel.parquet, sources/vr_panel.parquet
  sources/raw_tickers/*.parquet   raw daily OHLCV for 35 tickers (yfinance/Bloomberg,
                                  pre-remediation); the panel's daily returns are
                                  rebuilt from them and checked against the cached PE.

Run: python3 revision_diagnostics.py   (deterministic; seeds fixed below)
"""
from math import factorial, log
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
DEV_END = pd.Timestamp("2021-12-31")
MIN_NAMES = 100
CLOSED = [pd.Timestamp(d) for d in ["2014-04-24", "2014-10-15", "2026-01-15",
                                    "2026-05-01", "2026-05-28", "2026-06-26"]]
LOW_TAIL = ["SHREEASHTA", "RCOM", "SUZLON", "NEUEON", "GTLINFRA", "RPOWER", "SKUMARSYNF",
            "SPTL", "IBULLSLTD", "JPPOWER", "GBGLOBAL", "INDUSINDBK", "RELIGARE",
            "AKSHOPTFBR", "TV18BRDCST"]   # the 15 tickers with the lowest minimum PE, 2011-2021
# 20 tickers drawn with numpy default_rng(20261014).choice(sorted dev tickers, 20) - IBULLSLTD
# appears in both lists.
RANDOM = ["KEI", "MAHSEAMLES", "GUJRATGAS", "NAVA", "J&KBANK", "OMAXE", "CEMPRO", "RAIN",
          "UPL", "ROUTE", "IBULLSLTD", "KRBL", "GRUH", "SUVEN", "BBOX", "JAMNAAUTO",
          "LALPATHLAB", "ALBK", "MHRIL", "MCX"]
stats = []


def record(k, v, d):
    stats.append((k, v, d))


def zc(s):
    s = pd.Series(s, dtype=float); mu, sd = s.mean(), s.std(ddof=1)
    return (s - mu) / sd if sd > 0 else s * 0.0


def newey_west(x, lag=6):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; T = len(x)
    mu = x.mean(); e = x - mu; var = (e @ e) / T
    for L in range(1, lag + 1):
        var += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / T
    se = np.sqrt(var / T)
    return mu, se, mu / se, T


def patterns(x, m=4):
    """Ordinal patterns exactly as the M7 estimator forms them (np.argsort on each window)."""
    return [tuple(np.argsort(x[i:i + m])) for i in range(len(x) - m + 1)]


def pe_from(pats, m=4):
    if not pats:
        return np.nan
    _, c = np.unique(np.array(pats), axis=0, return_counts=True)
    p = c / c.sum()
    return -(p * np.log(p)).sum() / log(factorial(m))


# ---------------------------------------------------------------------------
# A. Tie / stale-price diagnostics on 35 tickers (rebuilt daily returns)
# ---------------------------------------------------------------------------
e = pd.read_parquet(SRC / "entropy_panel.parquet")
dev = e[e.rb <= DEV_END]
rng = np.random.default_rng(20261014)
rows = []
for name in sorted(set(LOW_TAIL) | set(RANDOM)):
    r = pd.read_parquet(SRC / "raw_tickers" / f"{name}_NS.parquet").sort_values("Date")
    r = r[~r.Date.isin(CLOSED)].reset_index(drop=True)
    phantom = (r.Open == r.High) & (r.High == r.Low) & (r.Low == r.Close) & (r.Volume == 0)
    close = r.Close.where(~phantom)
    ret = (close / close.shift(1) - 1).values
    dates = r.Date.values
    tk = name + ".NS"
    for _, row in dev[dev.ticker == tk].iterrows():
        p = int(np.searchsorted(dates, np.datetime64(row.rb)))
        w = ret[p - 252:p - 21 + 1]
        w = w[~np.isnan(w)]
        pats = patterns(w)
        tied = [len(set(w[i:i + 4])) < 4 for i in range(len(w) - 3)]
        pe_orig = pe_from(pats)
        # (a) random tie-breaking: tiny jitter, averaged over 20 draws
        jit = [pe_from(patterns(w + rng.uniform(-1e-12, 1e-12, len(w)))) for _ in range(20)]
        # (b) tie-free windows only
        pe_tiefree = pe_from([pt for pt, t in zip(pats, tied) if not t])
        # (c) drop zero-return days before forming patterns
        wz = w[w != 0]
        pe_nozero = pe_from(patterns(wz))
        px = close.values[p - 252:p - 21 + 1]
        rows.append(dict(ticker=tk, rb=row.rb, group="low_tail" if name in LOW_TAIL else "random",
                         in_random=name in RANDOM, n_ret=len(w), pe_cached=row.pe4,
                         pe_rebuilt=pe_orig, zero_share=np.mean(w == 0),
                         tied_window_share=np.mean(tied), any_tie=bool(np.any(tied)),
                         pe_jitter=np.mean(jit), pe_tiefree=pe_tiefree, pe_nozero=pe_nozero,
                         median_price=np.nanmedian(px)))
td = pd.DataFrame(rows)
td.to_csv(HERE / "revision_tie_detail.csv", index=False)
record("tie_rebuild_rows", len(td), "stock-months rebuilt from raw files (35 tickers, 2011-2021)")
record("tie_rebuild_exact", int((np.abs(td.pe_rebuilt - td.pe_cached) < 1e-9).sum()),
       "of which rebuilt PE equals cached PE to 1e-9")
for g, sub in [("low", td[td.group == "low_tail"]), ("rand", td[td.in_random])]:
    record(f"tie_{g}_n", len(sub), f"stock-months, {g} group")
    record(f"tie_{g}_tickers", sub.ticker.nunique(), f"tickers, {g} group")
    record(f"tie_{g}_any_tie_share", sub.any_tie.mean(), "share of stock-months with >=1 tied 4-window")
    record(f"tie_{g}_tied_window_med", sub.tied_window_share.median(), "median share of tied 4-windows")
    record(f"tie_{g}_zero_med", sub.zero_share.median(), "median share of zero-return days")
    record(f"tie_{g}_pe_med", sub.pe_cached.median(), "median cached PE")
    record(f"tie_{g}_jit_shift_med", (sub.pe_jitter - sub.pe_cached).median(), "median PE change, random tie-break")
    record(f"tie_{g}_nozero_shift_med", (sub.pe_nozero - sub.pe_cached).median(), "median PE change, zero days dropped")
    record(f"tie_{g}_tiefree_shift_med", (sub.pe_tiefree - sub.pe_cached).median(), "median PE change, tie-free windows only")
low = td[td.group == "low_tail"]
sub95 = low[low.pe_cached < 0.95]
record("tie_sub95_n", len(sub95), "low-tail stock-months with cached PE < 0.95 in subset")
record("tie_sub95_jit_ge095", (sub95.pe_jitter >= 0.95).mean(), "share of those reaching PE>=0.95 under random tie-break")
record("tie_sub95_zero_med", sub95.zero_share.median(), "median zero-return share, PE<0.95 stock-months")
record("tie_sub95_jit_min", sub95.pe_jitter.min(), "min PE under random tie-break, PE<0.95 stock-months")
s = td[(td.ticker == "SHREEASHTA.NS") & (td.rb == pd.Timestamp("2013-07-31"))].iloc[0]
for k in ["pe_cached", "pe_jitter", "pe_nozero", "pe_tiefree", "zero_share",
          "tied_window_share", "median_price", "n_ret"]:
    record(f"shree_{k}", s[k], f"SHREEASHTA 2013-07-31: {k}")
record("dev_pe_lt095_rows", int((dev.pe4 < 0.95).sum()), "dev stock-months with PE<0.95 (full panel)")
record("dev_pe_lt095_tickers", int(dev[dev.pe4 < 0.95].ticker.nunique()), "tickers with any PE<0.95")
covered = dev[dev.pe4 < 0.95].ticker.isin([t + ".NS" for t in LOW_TAIL]).mean()
record("dev_pe_lt095_covered", covered, "share of PE<0.95 stock-months whose ticker is in the inspected low-tail set")

fig, ax = plt.subplots(figsize=(6.2, 2.6))
bins = np.linspace(0.80, 1.0, 81)
ax.hist(low.pe_cached, bins=bins, histtype="step", color="black", lw=1.4, label="low-tail tickers, as computed (ties by sort order)")
ax.hist(low.pe_jitter, bins=bins, histtype="step", color="0.45", lw=1.4, ls="--", label="low-tail tickers, ties broken at random")
ax.hist(td[td.in_random].pe_cached, bins=bins, color="0.8", label="random 20 tickers, as computed")
ax.set_xlabel("normalized permutation entropy, m = 4 (2011-2021 stock-months)")
ax.set_ylabel("count")
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "figures" / "fig4_ties.pdf"); plt.close(fig)

# ---------------------------------------------------------------------------
# B. M7 gate-1 sensitivity to the low tail (cached panel; feasible without raw data)
# ---------------------------------------------------------------------------
def m7_gate(df, transform="z", drop_below=None):
    unc, con = [], []
    for rb, g in df.groupby("rb"):
        d = g.dropna(subset=["pe4", "mom", "dgw", "vol60", "fwd"]).copy()
        if drop_below is not None:
            d = d[d.pe4 >= drop_below]
        if len(d) < MIN_NAMES:
            continue
        x = zc(d.pe4) if transform == "z" else zc(d.pe4.rank())
        zm, zd = zc(d.mom), zc(d.dgw)
        unc.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), x, zm]), d.fwd.values, rcond=None)[0][1])
        con.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), x, zm, zd]), d.fwd.values, rcond=None)[0][1])
    return newey_west(unc), newey_west(con)
for tag, df in [("full", e), ("dev", dev)]:
    for spec, kw in [("drop095", dict(drop_below=0.95)), ("rank", dict(transform="rank"))]:
        (mu_u, _, t_u, T), (mu_c, _, t_c, _) = m7_gate(df, **kw)
        record(f"m7_{tag}_{spec}_unc", mu_u, f"M7 uncontrolled coef, {spec}, {tag}")
        record(f"m7_{tag}_{spec}_unc_t", t_u, f"NW t, {spec}, {tag}")
        record(f"m7_{tag}_{spec}_dgw", mu_c, f"M7 +DGW coef, {spec}, {tag}")
        record(f"m7_{tag}_{spec}_dgw_t", t_c, f"NW t, {spec}, {tag}")
        record(f"m7_{tag}_{spec}_T", T, f"months, {spec}, {tag}")
pe_z_min = dev.groupby("rb").apply(lambda g: ((g.pe4 - g.pe4.mean()) / g.pe4.std()).min(), include_groups=False).min()
record("pe_min_z", pe_z_min, "most extreme within-month z-score of PE, dev")

# ---------------------------------------------------------------------------
# C. Test D strict development sample: drop the 2021-12-31 rebalance
# ---------------------------------------------------------------------------
vr = pd.read_parquet(SRC / "vr_panel.parquet")
def d_gate(df):
    unc, res = [], []
    for rb, g in df.groupby("rb"):
        d = g.dropna(subset=["vr", "rv", "mom", "fwd"]).copy()
        if len(d) < MIN_NAMES:
            continue
        zvr, zrv, zmom = zc(d.vr), zc(np.log(d.rv.clip(lower=1e-9))), zc(d.mom)
        zam = zc(np.log(d.amihud.clip(lower=1e-12)))
        b = np.polyfit(zrv, zvr, 1); u = zvr - (b[0] * zrv + b[1])
        unc.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), zvr]), d.fwd.values, rcond=None)[0][1])
        res.append(np.linalg.lstsq(np.column_stack([np.ones(len(d)), u, zrv, zmom, zam]), d.fwd.values, rcond=None)[0][1])
    return newey_west(unc), newey_west(res)
strict = vr[vr.rb < pd.Timestamp("2021-12-31")]
(mu_u, se_u, t_u, T), (mu_r, se_r, t_r, _) = d_gate(strict)
record("d_strict_rows", len(strict), "VR stock-months after dropping 2021-12-31")
record("d_strict_last_rb", str(strict.rb.max().date()), "last rebalance in strict sample")
record("d_strict_T", T, "months used")
record("d_strict_unc", mu_u, "D uncontrolled coef, strict"); record("d_strict_unc_t", t_u, "NW t")
record("d_strict_res", mu_r, "D gate coef, strict"); record("d_strict_res_t", t_r, "NW t")
record("d_strict_res_ci_lo", mu_r - 1.96 * se_r, "95% NW interval lower")
record("d_strict_res_ci_hi", mu_r + 1.96 * se_r, "95% NW interval upper")

# ---------------------------------------------------------------------------
# D. VR random-walk probe: documented reruns at two lengths
# ---------------------------------------------------------------------------
def variance_ratio(r, q):
    r = np.asarray(r, float); n = len(r); mu = r.mean(); s1 = np.sum((r - mu) ** 2) / (n - 1)
    X = np.concatenate([[0.0], np.cumsum(r)]); dq = (X[q:] - X[:-q]) - q * mu
    return (np.sum(dq ** 2) / (q * (n - q + 1) * (1 - q / n))) / s1
for n in (232, 252):
    for nseeds in (200, 20000):
        v = np.array([variance_ratio(np.random.default_rng(s).standard_normal(n), 21) for s in range(nseeds)])
        record(f"vrprobe_n{n}_s{nseeds}_mean", v.mean(), f"RW VR(21) mean, Gaussian returns, n={n}, seeds 0..{nseeds-1}")
        record(f"vrprobe_n{n}_s{nseeds}_sd", v.std(ddof=1), f"RW VR(21) SD, n={n}, seeds 0..{nseeds-1}")
        record(f"vrprobe_n{n}_s{nseeds}_median", np.median(v), f"RW VR(21) median, n={n}, seeds 0..{nseeds-1}")

# ---------------------------------------------------------------------------
# E. Skipped / thin months (calendar-defect inventory) and low-tail price profile
# ---------------------------------------------------------------------------
for nm, df in [("pe", e), ("vr", vr)]:
    per = df.groupby(df.rb.dt.to_period("M")).size()
    full = pd.period_range(per.index.min(), per.index.max(), freq="M")
    record(f"{nm}_missing_months", ";".join(str(m) for m in full if m not in per.index), "calendar months with no rebalance row")
    record(f"{nm}_thin_months", ";".join(f"{k}:{v}" for k, v in per[per < MIN_NAMES].items()), "months with <100 rows (dropped)")
g = e.dropna(subset=["pe4", "mom", "dgw", "vol60", "fwd"]).groupby("rb").size()
record("pe_full_dropped_after_na", ";".join(f"{k.date()}:{v}" for k, v in g[g < MIN_NAMES].items()), "M7 full-sample months dropped after NaN filter")
cg = pd.read_csv(sorted(SRC.glob("c_gate_monthly_correlations_*.csv"))[0])
fullc = pd.period_range("2010-06", "2021-12", freq="M")
record("c_missing_months", ";".join(str(m) for m in fullc if str(m) not in set(cg.month)), "Test C months absent, 2010-06..2021-12")
record("tie_sub95_price_lt10", (sub95.median_price < 10).mean(), "share of PE<0.95 subset stock-months with window median price < 10 rupees")
record("tie_sub95_price_med", sub95.median_price.median(), "median window price, PE<0.95 subset")
record("tie_rand_price_med", td[td.in_random].median_price.median(), "median window price, random subset")

out = pd.DataFrame(stats, columns=["key", "value", "description"])
out.to_csv(HERE / "revision_stats.csv", index=False)
print(out.to_string(index=False))
