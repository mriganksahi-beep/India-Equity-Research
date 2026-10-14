"""Round-2 revision diagnostics for the models-vs-rules paper (POST HOC; run by the author).

Why this script exists: the inputs below are larger than the 10 MB limit of the
Drive connector used to draft the paper, so these computations could not be run
while drafting. Everything here is a post hoc revision diagnostic. None of it was
part of the original pre-committed Phase 2/3 analysis.

Inputs (read-only):
  --preds  phase2_oos_predictions_2026-10-03.parquet   (Drive 1_fveQUkSz7HKeIvpPSM0z-PbogJk1XkL)
  --v1     ml_sandbox_v1_dataset_2026-09-07.parquet     (Drive 109vFoEet_KRU-G-p0X39tyzO7DR7jcU3), optional

What it does:
  1. GATE: rebuild the per-date log-loss/AUC table exactly as the Phase 2 notebook
     did, rerun the ORIGINAL i.i.d. calendar-month bootstrap (seed 0, B=1000), and
     check that the published point estimates and 95% CIs are reproduced. Any
     mismatch stops the script.
  2. Moving-block bootstrap over calendar months (block 3 primary; 2 and 6 as
     sensitivity). The same resampled months are used for M1, M2 and the
     M2-minus-M1 gain (paired), and for skill and AUC. 95% and 98.75% (4 events)
     percentile intervals. The pre-committed rules 1 and 2 are re-applied.
  3. (needs --v1) High-vol baseline: per-date AUC of ranking stocks by current
     realized_vol_60 (no fitting) for each event, versus the classifier's AUC, with
     a paired block-3 interval on the difference.
  4. (needs --v1) Feature geometry over ALL dates: distribution of per-date pairwise
     correlations of the z-scored features, and per-date PCA variance shares.
  5. (needs --v1) Log-turnover drift: yearly quantiles of raw log_turnover and of
     its within-date z-score.

Usage (Colab):
  python rerun_phase2_bootstrap.py --preds /content/drive/MyDrive/moonshot_ml_sandbox/phase2_oos_predictions_2026-10-03.parquet \
      --v1 /content/drive/MyDrive/moonshot_ml_sandbox/ml_sandbox_v1_dataset_2026-09-07.parquet --out ./revision_out
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
RUNGS = ["M0", "M1", "M2"]
EVENTS = ["ev_outperform", "ev_top", "ev_crash", "ev_highvol"]
RAW = ["momentum_12_1", "reversal_1m", "realized_vol_60", "log_turnover", "overnight_drift"]
Z = [f + "_z" for f in RAW]

# Published Phase 2 values (moonshot_ml_sandbox_phase2_logistic.ipynb, summary cell).
PUBLISHED = {
    "ev_outperform": dict(M1_skill=0.0027, M1_skill_lo=0.0002, M1_skill_hi=0.0052, M1_auc=0.5371,
                          M1_auc_lo=0.5255, M1_auc_hi=0.5493, gain=-0.0001, gain_lo=-0.0004, gain_hi=0.0002),
    "ev_top": dict(M1_skill=0.0022, M1_skill_lo=0.0006, M1_skill_hi=0.0037, M1_auc=0.5366,
                   M1_auc_lo=0.5274, M1_auc_hi=0.5459, gain=0.0003, gain_lo=-0.0003, gain_hi=0.0009),
    "ev_crash": dict(M1_skill=0.0192, M1_skill_lo=0.0134, M1_skill_hi=0.0245, M1_auc=0.5957,
                     M1_auc_lo=0.5820, M1_auc_hi=0.6086, gain=0.0013, gain_lo=0.0005, gain_hi=0.0021),
    "ev_highvol": dict(M1_skill=0.1292, M1_skill_lo=0.1205, M1_skill_hi=0.1378, M1_auc=0.7583,
                       M1_auc_lo=0.7508, M1_auc_hi=0.7653, gain=0.0143, gain_lo=0.0117, gain_hi=0.0170),
}
TOL_POINT, TOL_CI = 1e-4, 1e-4  # published values are rounded to 4 decimals


# ---- copied verbatim (logic) from the Phase 2 engine -------------------------------
def logloss_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def daily_table(dates, y, preds):
    d = pd.DataFrame({"date": dates, "y": y})
    out = {}
    g = d.groupby("date")["y"]
    n = g.size(); n1 = g.sum(); n0 = n - n1
    out["n"] = n
    for r, p in preds.items():
        d["ll"] = logloss_vec(d["y"].to_numpy(), p)
        out[f"ll_{r}"] = d.groupby("date")["ll"].sum()
        d["rk"] = pd.Series(p, index=d.index).groupby(d["date"]).rank(method="average")
        sr = d.loc[d["y"] == 1].groupby("date")["rk"].sum().reindex(n.index).fillna(0.0)
        auc = (sr - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc[(n1 == 0) | (n0 == 0)] = np.nan
        out[f"auc_{r}"] = auc
    return pd.DataFrame(out)


def summarize(daily, rung):
    return 1 - daily[f"ll_{rung}"].sum() / daily["ll_M0"].sum(), daily[f"auc_{rung}"].mean()


def month_aggregate(daily):
    month = pd.DatetimeIndex(daily.index).to_period("M")
    agg = pd.DataFrame({
        "ll0": daily["ll_M0"], "ll1": daily["ll_M1"], "ll2": daily["ll_M2"],
        "a1": daily["auc_M1"].fillna(0), "a2": daily["auc_M2"].fillna(0),
        "na": daily["auc_M1"].notna().astype(float)}).groupby(month).sum()
    return agg


def stats_from_counts(C, A):
    T = C @ A
    ll0, ll1, ll2, a1, a2, na = T.T
    s1, s2 = 1 - ll1 / ll0, 1 - ll2 / ll0
    return dict(skill_M1=s1, skill_M2=s2, gain=s2 - s1, auc_M1=a1 / na, auc_M2=a2 / na)


def iid_month_bootstrap(daily, B=1000, seed=0):
    """The ORIGINAL Phase 2 bootstrap: calendar months drawn i.i.d. with replacement."""
    rng = np.random.default_rng(seed)
    A = month_aggregate(daily).to_numpy(); M = len(A)
    idx = rng.integers(0, M, size=(B, M))
    C = np.zeros((B, M))
    for b in range(B):
        C[b] = np.bincount(idx[b], minlength=M)
    return stats_from_counts(C, A)


def moving_block_indices(M, L, B, seed):
    """Moving-block bootstrap over M consecutive calendar months, block length L."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(M / L))
    out = np.empty((B, M), dtype=int)
    for b in range(B):
        starts = rng.integers(0, M - L + 1, size=nb)
        out[b] = np.concatenate([np.arange(s, s + L) for s in starts])[:M]
    return out


def counts_from_indices(idx, M):
    C = np.zeros((idx.shape[0], M))
    for b in range(idx.shape[0]):
        C[b] = np.bincount(idx[b], minlength=M)
    return C


def pct(x, level):
    a = (100 - level) / 2
    return np.percentile(x, [a, 100 - a])


# ---- main analysis ------------------------------------------------------------------
def phase2_tables(preds):
    tabs = {}
    for e in EVENTS:
        p = preds[preds["event"] == e]
        dl = daily_table(p["date"].to_numpy(), p["y"].to_numpy(),
                         {r: p[f"p_{r}"].to_numpy() for r in RUNGS})
        dl.index = pd.DatetimeIndex(dl.index)
        dl["test_year"] = dl.index.year
        tabs[e] = dl
    return tabs


def gate_reproduce(tabs):
    rows, ok = [], True
    for e, dl in tabs.items():
        bt = iid_month_bootstrap(dl, B=1000, seed=0)
        s1, a1 = summarize(dl, "M1"); s2, _ = summarize(dl, "M2")
        got = dict(M1_skill=s1, M1_skill_lo=pct(bt["skill_M1"], 95)[0], M1_skill_hi=pct(bt["skill_M1"], 95)[1],
                   M1_auc=a1, M1_auc_lo=pct(bt["auc_M1"], 95)[0], M1_auc_hi=pct(bt["auc_M1"], 95)[1],
                   gain=s2 - s1, gain_lo=pct(bt["gain"], 95)[0], gain_hi=pct(bt["gain"], 95)[1])
        for k, v in got.items():
            ref = PUBLISHED[e][k]
            tol = TOL_POINT if k in ("M1_skill", "M1_auc", "gain") else TOL_CI
            match = abs(round(v, 4) - ref) <= tol + 1e-12
            ok &= match
            rows.append(dict(event=e, quantity=k, published=ref, recomputed=v, match=match))
    return pd.DataFrame(rows), ok


def block_rerun(tabs, blocks=(2, 3, 6), B=1000, seed=20261014):
    rows = []
    for e, dl in tabs.items():
        A = month_aggregate(dl).to_numpy(); M = len(A)
        s1, a1 = summarize(dl, "M1"); s2, a2 = summarize(dl, "M2")
        early, late = dl[dl.test_year <= 2019], dl[dl.test_year >= 2020]
        s1e, s1l = summarize(early, "M1")[0], summarize(late, "M1")[0]
        g_e = summarize(early, "M2")[0] - s1e; g_l = summarize(late, "M2")[0] - s1l
        for L in blocks:
            idx = moving_block_indices(M, L, B, seed + L)        # shared by every quantity (paired)
            bt = stats_from_counts(counts_from_indices(idx, M), A)
            r = dict(event=e, block_months=L, B=B, seed=seed + L, n_months=M,
                     M1_skill=s1, M1_auc=a1, M2_skill=s2, M2_auc=a2, gain=s2 - s1,
                     skill_early=s1e, skill_late=s1l, gain_early=g_e, gain_late=g_l)
            for lev in (95, 98.75):
                tag = "95" if lev == 95 else "9875"
                for q in ("skill_M1", "auc_M1", "skill_M2", "auc_M2", "gain"):
                    lo, hi = pct(bt[q], lev)
                    r[f"{q}_lo{tag}"], r[f"{q}_hi{tag}"] = lo, hi
            r["rule1"] = bool(r["skill_M1_lo95"] > 0 and r["auc_M1_lo95"] > 0.5 and s1e > 0 and s1l > 0)
            r["rule2"] = bool(r["gain_lo95"] > 0 and (s2 - s1) > 0.002 and g_e > 0 and g_l > 0)
            rows.append(r)
    return pd.DataFrame(rows)


def vol_baseline(preds, v1, B=1000, L=3, seed=314):
    """Per-date AUC of ranking by current realized_vol_60 (higher = more likely event)."""
    v = v1[["date", "ticker", "realized_vol_60"]]
    out = []
    for e in EVENTS:
        rung = "M2" if e == "ev_highvol" else "M1"
        p = preds[preds["event"] == e].merge(v, on=["date", "ticker"], how="left", validate="one_to_one")
        miss = p["realized_vol_60"].isna().sum()
        if miss:
            raise ValueError(f"{e}: {miss} prediction rows lack realized_vol_60 in V1")
        dl = daily_table(p["date"].to_numpy(), p["y"].to_numpy(),
                         {"M0": p["p_M0"].to_numpy(), "MOD": p[f"p_{rung}"].to_numpy(),
                          "VOL": p["realized_vol_60"].to_numpy()})
        dl.index = pd.DatetimeIndex(dl.index)
        month = dl.index.to_period("M")
        agg = pd.DataFrame({"am": dl["auc_MOD"].fillna(0), "av": dl["auc_VOL"].fillna(0),
                            "na": dl["auc_MOD"].notna().astype(float)}).groupby(month).sum().to_numpy()
        idx = moving_block_indices(len(agg), L, B, seed)
        T = counts_from_indices(idx, len(agg)) @ agg
        diff = T[:, 0] / T[:, 2] - T[:, 1] / T[:, 2]
        lo, hi = pct(diff, 95)
        out.append(dict(event=e, model_rung=rung, auc_model=dl["auc_MOD"].mean(),
                        auc_current_vol=dl["auc_VOL"].mean(),
                        diff=dl["auc_MOD"].mean() - dl["auc_VOL"].mean(), diff_lo95=lo, diff_hi95=hi,
                        block_months=L, B=B, seed=seed))
    return pd.DataFrame(out)


def feature_geometry(v1, min_names=30):
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    corr_rows, pca_rows = [], []
    for d, day in v1.groupby("date"):
        if len(day) < min_names:
            continue
        X = day[Z].to_numpy(float)
        C = np.corrcoef(X, rowvar=False)
        corr_rows.append({f"{RAW[i]}|{RAW[j]}": C[i, j] for i, j in pairs} | {"date": d})
        Xc = X - X.mean(0)
        ev = np.sort(np.linalg.eigvalsh(np.cov(Xc, rowvar=False)))[::-1]
        pca_rows.append({f"PC{k+1}": ev[k] / ev.sum() for k in range(5)} | {"date": d})
    corr = pd.DataFrame(corr_rows).set_index("date")
    pca = pd.DataFrame(pca_rows).set_index("date")
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    return corr, pca, corr.quantile(q).T, pca.quantile(q).T, corr.mean().rename("mean")


def turnover_drift(v1):
    g = v1.groupby(v1["date"].dt.year)
    q = lambda c, x: g[c].quantile(x)
    return pd.DataFrame({"n_cards": g.size(),
                         "raw_p10": q("log_turnover", .1), "raw_median": q("log_turnover", .5),
                         "raw_p90": q("log_turnover", .9),
                         "z_p10": q("log_turnover_z", .1), "z_median": q("log_turnover_z", .5),
                         "z_p90": q("log_turnover_z", .9)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--v1", default=None)
    ap.add_argument("--out", default="revision_out")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--skip-gate", action="store_true", help="only for synthetic self-tests")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(a.preds)
    preds["date"] = pd.to_datetime(preds["date"])
    tabs = phase2_tables(preds)

    gate, ok = gate_reproduce(tabs)
    gate.to_csv(out / "gate_iid_reproduction.csv", index=False)
    print(gate.to_string(index=False))
    if not ok and not a.skip_gate:
        raise SystemExit("GATE FAILED: the original i.i.d. intervals were not reproduced; stop.")
    print("[GATE PASS] original Phase 2 point estimates and i.i.d. CIs reproduced")

    blk = block_rerun(tabs, B=a.B)
    blk.to_csv(out / "phase2_moving_block.csv", index=False)
    cols = ["event", "block_months", "M1_skill", "skill_M1_lo95", "skill_M1_hi95", "M1_auc",
            "auc_M1_lo95", "auc_M1_hi95", "gain", "gain_lo95", "gain_hi95", "rule1", "rule2"]
    print(blk[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    summary = {"gate_pass": bool(ok), "B": a.B, "blocks": [2, 3, 6]}

    if a.v1:
        v1 = pd.read_parquet(a.v1)
        v1["date"] = pd.to_datetime(v1["date"])
        vb = vol_baseline(preds, v1, B=a.B)
        vb.to_csv(out / "vol_baseline_auc.csv", index=False)
        print(vb.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        corr, pca, cq, pq, cmean = feature_geometry(v1)
        corr.to_csv(out / "per_date_correlations.csv"); pca.to_csv(out / "per_date_pca_shares.csv")
        cq.to_csv(out / "per_date_correlation_quantiles.csv"); pq.to_csv(out / "per_date_pca_quantiles.csv")
        print(cq.round(3).to_string()); print(pq.round(3).to_string())
        print("mean of per-date correlations (should match the visualization notebook [S3]):")
        print(cmean.round(3).to_string())
        td = turnover_drift(v1)
        td.to_csv(out / "log_turnover_by_year.csv")
        print(td.round(3).to_string())
        summary["n_dates_geometry"] = int(len(corr))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
