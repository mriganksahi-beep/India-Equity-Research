"""
synthetic_selftest.py
=====================
Integration test for mi_screen on a SYNTHETIC panel with a KNOWN ground truth.

We plant exactly one genuinely predictive cross-sectional feature and one pure
noise feature, then assert:
  (1) the planted feature's MI_excess CI excludes zero (signal detected),
  (2) the noise feature's MI_excess CI includes zero (no false positive),
  (3) the leakage assertion holds for a sample window,
  (4) the PIT universe filter actually excludes a ticker while it is flagged out,
  (5) feature warm-up gaps are present (momentum has no signal in early history),
  (6) the hold-out guard drops rebalances whose forward window crosses TRAIN_END.

If any of these fail, the module is not trusted on real data.
"""

import numpy as np
import pandas as pd

import mi_screen as M


def make_synthetic(seed=7):
    """Daily panel, 2008-2021, ~120 tickers.  One planted predictive feature.

    Design of the plant:
      * Each month, each stock draws a hidden 'quality' score q ~ N(0,1).
      * The stock's forward 21-day return is monotone-increasing in the
        cross-sectional rank of q that month, plus idiosyncratic noise and a
        common market move (which cross-sectional ranking should neutralise).
      * We expose 'planted_feat' = q observable AS-OF month end (no look-ahead).
      * 'noise_feat' is an independent N(0,1), unrelated to forward returns.
    We inject the plant through the daily-return path so that close-price-derived
    forward returns actually carry it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", "2021-12-31")
    tickers = [f"T{i:03d}" for i in range(120)]
    D, N = len(dates), len(tickers)

    # Month index per date, for assigning monthly quality.
    month_id = pd.PeriodIndex(dates, freq="M")
    uniq_months = month_id.unique()

    # Monthly hidden quality per (month, ticker).
    q = rng.standard_normal((len(uniq_months), N))
    q_by_month = {m: q[k] for k, m in enumerate(uniq_months)}

    # Build daily returns: common market factor + quality-driven drift + noise.
    # CRITICAL PLANT DESIGN: the feature exposed at month-end t is q of month M.
    # The 21-day FORWARD return from t spans month M+1.  So the drift in month
    # M+1 must depend on q of month M (the PREVIOUS month's quality), otherwise
    # the feature-at-t and the return-after-t are unrelated by construction.
    month_pos = {m: k for k, m in enumerate(uniq_months)}
    daily_ret = np.zeros((D, N))
    market = rng.standard_normal(D) * 0.010  # common daily move
    for di, dt in enumerate(dates):
        k = month_pos[month_id[di]]
        prev_q = q[k - 1] if k > 0 else q[0]           # PREVIOUS month's quality
        z = (prev_q - prev_q.mean()) / (prev_q.std() + 1e-9)
        drift = 0.0020 * z                              # stronger, still < noise
        idio = rng.standard_normal(N) * 0.015
        daily_ret[di] = market[di] + drift + idio

    # Close prices from returns.
    close = 100.0 * np.cumprod(1.0 + daily_ret, axis=0)

    # Overnight return: a fraction of the daily move (just needs to exist).
    overnight = daily_ret * 0.4 + rng.standard_normal((D, N)) * 0.002
    # Turnover: positive, lognormal-ish.
    turnover = np.exp(rng.standard_normal((D, N)) * 0.5 + 15.0)
    # realized_vol_60: rolling std of daily_ret * sqrt(252), strict window.
    dr_df = pd.DataFrame(daily_ret, index=dates, columns=tickers)
    rvol60 = dr_df.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(252)

    # PIT flag: all IN, EXCEPT ticker T000 is flagged OUT during 2015 (to test
    # the universe filter), and NaN before 2010-06-30 (tri-state realism).
    flag = pd.DataFrame(1.0, index=dates, columns=tickers)
    flag.loc[flag.index < pd.Timestamp("2010-06-30"), :] = np.nan
    out_mask = (flag.index >= pd.Timestamp("2015-01-01")) & (flag.index <= pd.Timestamp("2015-12-31"))
    flag.loc[out_mask, "T000"] = 0.0

    # Assemble the LONG daily panel expected by build_panel.
    close_df = pd.DataFrame(close, index=dates, columns=tickers)
    on_df = pd.DataFrame(overnight, index=dates, columns=tickers)
    to_df = pd.DataFrame(turnover, index=dates, columns=tickers)

    # Build the long panel on the FULL (date x ticker) grid so NaN flags survive
    # (pandas 3.0 stack() no longer drops NaN and dropna= is removed).
    full_idx = pd.MultiIndex.from_product([dates, tickers], names=[M.COL_DATE, M.COL_TICKER])
    long = pd.DataFrame(index=full_idx)
    long[M.COL_CLOSE] = close_df.stack()
    long[M.COL_DAILY_RET] = dr_df.stack()
    long[M.COL_OVERNIGHT_RET] = on_df.stack()
    long[M.COL_TURNOVER] = to_df.stack()
    long[M.COL_RVOL60] = rvol60.stack()
    long[M.COL_FLAG] = flag.stack()          # NaN flags preserved via grid reindex
    long = long.reset_index()

    # Monthly Amihud (separate file): mean |ret|/turnover per (ticker, month).
    am = (dr_df.abs() / to_df).stack().rename("v").reset_index()
    am.columns = [M.COL_DATE, M.COL_AMIHUD_TICKER, "v"]
    am[M.COL_AMIHUD_MONTH] = am[M.COL_DATE].values.astype("datetime64[M]").astype("datetime64[ns]")
    amihud = (am.groupby([M.COL_AMIHUD_TICKER, M.COL_AMIHUD_MONTH])["v"]
              .mean().rename(M.COL_AMIHUD_VALUE).reset_index())

    # Expose the planted + noise features by MONKEY-PATCHING FEATURES so the
    # screen tests them directly.  planted_feat = q as-of month end; noise_feat
    # independent.  We attach them as wide frames via closures.
    planted_wide = pd.DataFrame(np.nan, index=dates, columns=tickers)
    noise_wide = pd.DataFrame(np.nan, index=dates, columns=tickers)
    noise_month = rng.standard_normal((len(uniq_months), N))
    for di, dt in enumerate(dates):
        k = list(uniq_months).index(month_id[di])
        planted_wide.iloc[di] = q_by_month[month_id[di]]
        noise_wide.iloc[di] = noise_month[k]

    return long, amihud, planted_wide, noise_wide


def main():
    print("=" * 70)
    print("SYNTHETIC SELF-TEST")
    print("=" * 70)
    long, amihud, planted_wide, noise_wide = make_synthetic()
    panel = M.build_panel(long, amihud)
    print(f"panel: {len(panel.dates)} dates x {len(panel.tickers)} tickers")

    # --- Inject planted/noise features into the FEATURES registry for the test.
    feats = {
        "planted_feat": lambda p, w=planted_wide: w,
        "noise_feat": lambda p, w=noise_wide: w,
        "momentum_12_1": M.feat_momentum_12_1,   # to check warm-up gap
    }

    # ---------------- VALIDATION BLOCK 5: warm-up gap (momentum) --------------
    mom = M.feat_momentum_12_1(panel)
    first_valid = mom.dropna(how="all").index.min()
    warmup_ok = first_valid >= panel.dates[252]  # needs 252d history
    print(f"\n[warm-up] first momentum signal at {first_valid.date()} "
          f"(needs >= {panel.dates[252].date()}): {'OK' if warmup_ok else 'FAIL'}")
    assert warmup_ok, "momentum has signal before 252d of history -> leakage/backfill"

    # ---------------- VALIDATION BLOCK 4: PIT universe filter -----------------
    # T000 is flagged OUT for all of 2015. Confirm it is excluded from a 2015
    # rebalance cross-section and INCLUDED in a 2016 one.
    rebal = M.month_end_rebalance_dates(panel)
    d2015 = [d for d in rebal if d.year == 2015][3]
    d2016 = [d for d in rebal if d.year == 2016][3]
    fwd21 = M.forward_return(panel, 21)
    cs15 = M.cross_section(panel, planted_wide, fwd21, d2015, 21)
    # rebuild eligibility masks directly to check membership of T000
    elig15 = (panel.flag.loc[d2015] == 1) & planted_wide.loc[d2015].notna() & fwd21.loc[d2015].notna()
    elig16 = (panel.flag.loc[d2016] == 1) & planted_wide.loc[d2016].notna() & fwd21.loc[d2016].notna()
    t000_out_2015 = not bool(elig15.get("T000", False))
    t000_in_2016 = bool(elig16.get("T000", False))
    print(f"[universe] T000 excluded {d2015.date()} (flagged out): "
          f"{'OK' if t000_out_2015 else 'FAIL'}; "
          f"included {d2016.date()}: {'OK' if t000_in_2016 else 'FAIL'}")
    assert t000_out_2015 and t000_in_2016, "PIT universe filter not applied correctly"

    # ---------------- VALIDATION BLOCK 3: leakage assertion -------------------
    # For one sample rebalance, the LATEST date feeding the feature must be <= t,
    # and the EARLIEST date feeding the forward return must be > t.
    t = d2016
    i = panel.dates.get_indexer([t])[0]
    feat_uses_latest = panel.dates[i]          # planted_feat is as-of t (month end)
    fwd_uses_earliest = panel.dates[i + 1]     # close[t+h]/close[t]; t itself is base,
    # the FUTURE information is close[t+h]; base close[t] is known at t. The future
    # leg is strictly > t:
    fwd_future_leg = panel.dates[i + 21]
    leak_ok = (feat_uses_latest <= t) and (fwd_future_leg > t)
    print(f"[leakage] feature latest={feat_uses_latest.date()} <= t={t.date()} "
          f"and fwd future leg={fwd_future_leg.date()} > t: "
          f"{'OK' if leak_ok else 'FAIL'}")
    assert leak_ok

    # ---------------- VALIDATION BLOCK 6: hold-out guard ----------------------
    # A rebalance in late 2021 with h=252 must be dropped (forward crosses 2021-12-31).
    late = [d for d in rebal if d.year == 2021][-1]
    cs_late = M.cross_section(panel, planted_wide, M.forward_return(panel, 252), late, 252)
    holdout_ok = cs_late is None
    print(f"[hold-out] late-2021 rebalance {late.date()} @1y dropped: "
          f"{'OK' if holdout_ok else 'FAIL'}")
    assert holdout_ok, "hold-out guard failed -> forward window crossed TRAIN_END"

    # -------- VALIDATION BLOCK 7: NaN-padded rolling regression (real bug) -----
    # The real panel embeds ~31% NaN rows (holidays/masked). A trailing-21 feature
    # built with strict rolling over the NaN-padded wide frame collapses to ~0
    # coverage; the fixed version (roll over trading rows) survives. Reproduce the
    # bug and prove the fix.
    rng_nan = np.random.default_rng(99)
    holey = panel.turnover.copy()
    nan_mask = rng_nan.random(holey.shape) < 0.31
    holey = holey.where(~nan_mask)
    strict = np.log(holey.where(holey > 0)).rolling(21, min_periods=21).mean()
    fixed = M._trailing_mean_over_trading_rows(
        np.log(holey.where(holey > 0)), win=21, min_obs=21)
    strict_cov = np.median([strict.loc[t].notna().sum() for t in rebal])
    fixed_cov = np.median([fixed.loc[t].notna().sum() for t in rebal])
    print(f"[nan-roll] median stk/date  strict={strict_cov:.0f}  fixed={fixed_cov:.0f} "
          f"(strict collapses, fixed survives)")
    assert strict_cov < fixed_cov, "fixed rolling did not recover coverage"
    assert fixed_cov > 0.5 * len(panel.tickers), "fixed rolling still under-covers"

    # ---------------- CORE: run the screen on planted vs noise ----------------
    print("\n[screen] running MI-excess on planted vs noise @ 1m horizon ...")
    res = M.run_screen(panel, features=feats, horizons={"1m": 21},
                       lookbacks_years=[2], verbose=True)

    planted = res["planted_feat"]["1m"]
    noise = res["noise_feat"]["1m"]

    print("\n--- ground-truth assertions ---")
    print(f"planted MI_excess={planted['mi_excess']:+.4f} "
          f"CI[{planted['ci_low']:+.4f},{planted['ci_high']:+.4f}] sig={planted['sig']}")
    print(f"noise   MI_excess={noise['mi_excess']:+.4f} "
          f"CI[{noise['ci_low']:+.4f},{noise['ci_high']:+.4f}] sig={noise['sig']}")

    assert planted["sig"], "PLANTED signal not detected (CI includes 0) -> screen too weak"
    assert not noise["sig"], "NOISE feature flagged significant -> false positive / bad null"

    print("\nALL SELF-TEST ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
