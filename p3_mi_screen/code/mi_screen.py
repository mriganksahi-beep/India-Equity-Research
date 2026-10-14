"""
mi_screen.py
============
Point-in-time cross-sectional MUTUAL-INFORMATION screen for the
moonshot_compounding feature panel (NSE Nifty 500).

WHAT THIS MEASURES
------------------
For each (feature, forward-horizon) pair, how much information does the
CROSS-SECTIONAL ordering of the feature on a rebalance date carry about the
CROSS-SECTIONAL ordering of forward returns?  We report

    MI_excess = MI(feature_rank, fwd_ret_rank) - mean(MI under within-date shuffle)

i.e. estimated predictive information MINUS the noise floor that the finite-sample
KSG estimator reports even under independence.  A block-bootstrap CI over *dates*
tells us whether MI_excess is distinguishable from zero.

WHY RANKS
---------
Both feature and forward return are cross-sectionally rank-transformed to (0,1)
per rebalance date BEFORE pooling.  This (a) matches how cross-sectional signals
are actually traded (you trade the ranking), (b) removes the market factor
automatically -- a market-wide move leaves relative ranks unchanged, so no
separate demeaning is needed, (c) removes date-level scale differences so pooling
across dates is clean, and (d) lets MI catch NON-MONOTONE structure (e.g. a
vol/size hump) that rank-IC would miss -- that non-monotone sensitivity is the
main thing MI buys us over the IC we already compute.

CORRECTNESS STANCE
------------------
Every result is guilty (spurious) until proven innocent.  The module refuses to
run silently past a schema mismatch, a leakage violation, or a hold-out crossing.
It NEVER substitutes a value for an unknown one (fact: no interpolation /
forward-fill / zero-fill in the data layer).  Missing observations are DROPPED,
never zeroed.

Author note: this module is designed to be run in Colab against the real panel.
It ships with a synthetic self-test (see synthetic_selftest.py) that must pass
before it is trusted on real data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Training window.  Hold-out (2022-01-01 onward) is NEVER touched by this module:
# any rebalance whose forward-return window would cross TRAIN_END is dropped.
TRAIN_START = pd.Timestamp("2010-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")

# Forward horizons in TRADING days (~1w, 1m, 3m, 6m, 1y).
HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}

# Lookback windows for the stationarity probe, in calendar years.
LOOKBACKS_YEARS = [1, 2, 3]

# MI estimator settings.
KNN_K = 3                 # KSG neighbours (sklearn default 3)
N_SHUFFLE = 40            # within-date permutations for the noise floor
N_BOOT = 300              # block-bootstrap resamples over dates for the CI
CI_LOW, CI_HIGH = 5.0, 95.0
MIN_STOCKS_PER_DATE = 30  # a cross-section thinner than this is dropped, logged
MI_REDUNDANCY_HIGH = 0.30  # MI_excess above this between features is REPORTED but
                           # annotated: usually signals shared construction (e.g.
                           # Amihud's denominator is turnover), not a bug
RNG_SEED = 20260809

# Column names expected in the DAILY feature panel.  Asserted at load time so a
# schema drift fails loudly here, not with a KeyError deep in a loop.
# NOTE: matched to the REAL panel schema (capitalized OHLCV, 'Date').
COL_TICKER = "ticker"
COL_DATE = "Date"
COL_CLOSE = "Close"
COL_DAILY_RET = "daily_return"
COL_OVERNIGHT_RET = "overnight_return"
COL_TURNOVER = "turnover"
COL_RVOL60 = "realized_vol_60"
COL_FLAG = "in_nifty500_flag"          # PIT membership, tri-state (1/0/NaN)
COL_DATE_STATUS = "date_status"        # 'vendor_failure' etc. (fact #24/#25)

# Amihud lives in a SEPARATE monthly file (fact #19: never broadcast to daily).
COL_AMIHUD_TICKER = "ticker"
COL_AMIHUD_MONTH = "year_month"        # STRING like '2015-03' -> parsed to month-start
COL_AMIHUD_VALUE = "amihud_illiquidity"
COL_AMIHUD_EDGE = "is_edge_month"      # thin-coverage months, dropped by default


# ---------------------------------------------------------------------------
# SCHEMA VALIDATION
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{name}] missing required columns {missing}. "
            f"Present: {sorted(df.columns.tolist())}. "
            f"Fix COL_* constants at the top of mi_screen.py if the panel uses "
            f"different names -- do NOT let this pass silently."
        )


# ---------------------------------------------------------------------------
# WIDE FRAMES  (date x ticker)  -- makes every time shift positional & auditable
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    """Wide (date x ticker) frames.  Every feature and the forward return is a
    positional shift on these, so the point-in-time direction of every operation
    is visible in the code, not buried in a comment."""
    close: pd.DataFrame
    daily_ret: pd.DataFrame
    overnight_ret: pd.DataFrame
    turnover: pd.DataFrame
    rvol60: pd.DataFrame
    flag: pd.DataFrame
    amihud_monthly: pd.DataFrame  # long: [ticker, month, amihud]
    dates: pd.DatetimeIndex = field(init=False)
    tickers: pd.Index = field(init=False)
    date_status: pd.Series = field(init=False, default=None)  # per-date 'ok'/'vendor_failure'

    def __post_init__(self):
        self.dates = self.close.index
        self.tickers = self.close.columns
        # All daily frames must share the exact same axes -- otherwise a shift on
        # one and a read on another silently misalign.
        for nm, fr in [("daily_ret", self.daily_ret), ("overnight_ret", self.overnight_ret),
                       ("turnover", self.turnover), ("rvol60", self.rvol60), ("flag", self.flag)]:
            if not fr.index.equals(self.close.index):
                raise ValueError(f"{nm} date index does not match close.")
            if not fr.columns.equals(self.close.columns):
                raise ValueError(f"{nm} ticker columns do not match close.")


def build_panel(daily: pd.DataFrame, amihud_monthly: pd.DataFrame) -> Panel:
    """Pivot the long daily panel into aligned wide frames and attach Amihud."""
    _require_columns(
        daily,
        [COL_TICKER, COL_DATE, COL_CLOSE, COL_DAILY_RET, COL_OVERNIGHT_RET,
         COL_TURNOVER, COL_RVOL60, COL_FLAG],
        "daily panel",
    )
    _require_columns(
        amihud_monthly,
        [COL_AMIHUD_TICKER, COL_AMIHUD_MONTH, COL_AMIHUD_VALUE],
        "amihud monthly",
    )

    d = daily.copy()
    # Normalize Date: to datetime, strip any tz, force a single datetime64 unit.
    # fact #27: a datetime64[ns] vs [us] mismatch silently breaks date lookups.
    d[COL_DATE] = pd.to_datetime(d[COL_DATE])
    if getattr(d[COL_DATE].dt, "tz", None) is not None:
        d[COL_DATE] = d[COL_DATE].dt.tz_localize(None)
    d[COL_DATE] = d[COL_DATE].astype("datetime64[ns]")

    # Duplicate (ticker, date) rows would make pivot silently pick one -- forbid.
    dup = d.duplicated(subset=[COL_TICKER, COL_DATE]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (ticker,date) rows in daily panel; refuse to pivot.")

    def wide(col):
        return d.pivot(index=COL_DATE, columns=COL_TICKER, values=col).sort_index()

    close = wide(COL_CLOSE)
    panel = Panel(
        close=close,
        daily_ret=wide(COL_DAILY_RET).reindex_like(close),
        overnight_ret=wide(COL_OVERNIGHT_RET).reindex_like(close),
        turnover=wide(COL_TURNOVER).reindex_like(close),
        rvol60=wide(COL_RVOL60).reindex_like(close),
        flag=wide(COL_FLAG).reindex_like(close),
        amihud_monthly=amihud_monthly.copy(),
    )
    # Optional: per-date status (vendor_failure etc). A date is 'vendor_failure'
    # if ANY row that date carries it; store a per-date Series for driver asserts.
    if COL_DATE_STATUS in d.columns:
        ds = (d.groupby(COL_DATE)[COL_DATE_STATUS]
              .agg(lambda s: "vendor_failure" if (s == "vendor_failure").any() else "ok"))
        panel.date_status = ds.reindex(close.index)
    else:
        panel.date_status = pd.Series("ok", index=close.index)
    return panel


# ---------------------------------------------------------------------------
# REBALANCE GRID
# ---------------------------------------------------------------------------

def month_end_rebalance_dates(panel: Panel,
                              start: pd.Timestamp = TRAIN_START,
                              end: pd.Timestamp = TRAIN_END) -> pd.DatetimeIndex:
    """Last available TRADING date in each calendar month within [start, end].
    Uses the panel's own date index -- never a synthetic weekday calendar
    (fact: 2015-02-28 was a live Saturday session; weekday asserts false-fail)."""
    idx = panel.dates
    mask = (idx >= start) & (idx <= end)
    sub = idx[mask]
    if len(sub) == 0:
        raise ValueError("No trading dates in the requested rebalance range.")
    s = pd.Series(sub, index=sub)
    last = s.groupby([sub.year, sub.month]).max()
    return pd.DatetimeIndex(sorted(last.values))


# ---------------------------------------------------------------------------
# FEATURE CONSTRUCTION  (all observed AS-OF the rebalance date t)
# ---------------------------------------------------------------------------
# Each function returns a wide (date x ticker) frame whose value at row t uses
# ONLY information available at or before t.  The shift direction is the audit.

def feat_momentum_12_1(panel: Panel) -> pd.DataFrame:
    # Jegadeesh-Titman 12-1: trailing 12m return skipping the most recent 1m.
    # value@t = close[t-21] / close[t-252] - 1   (both strictly in the past)
    c = panel.close
    return c.shift(21) / c.shift(252) - 1.0


def feat_reversal_1m(panel: Panel) -> pd.DataFrame:
    # Short-term reversal: last-month return.  value@t = close[t]/close[t-21]-1.
    c = panel.close
    return c / c.shift(21) - 1.0


def feat_rvol60(panel: Panel) -> pd.DataFrame:
    # Already a trailing feature in the panel; read as-of t.
    return panel.rvol60


def _trailing_mean_over_trading_rows(wide: pd.DataFrame, win: int,
                                     min_obs: int) -> pd.DataFrame:
    """Trailing mean of a wide (date x ticker) frame, computed PER TICKER over
    that ticker's NON-NaN (trading) rows only, then reindexed back to all dates.

    WHY: the raw panel embeds masked/holiday/pre-listing rows as NaN (~31% of
    rebalance rows).  A naive `.rolling(win, min_periods=win)` over the wide frame
    needs `win` CONSECUTIVE non-NaN calendar rows, which almost never exist -> the
    feature is empty for the whole universe on most dates (median 0 stk/date).
    Rolling over trading rows only makes "trailing 21" mean 21 real observations,
    matching how realized_vol_60 was built in the pipeline.  value@t uses only
    observations at or before t (strictly backward)."""
    out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for c in wide.columns:
        s = wide[c]
        traded = s.dropna()
        if len(traded) < min_obs:
            continue
        rolled = traded.rolling(win, min_periods=min_obs).mean()
        out[c] = rolled.reindex(wide.index)  # NaN on non-trading rows, correct
    return out


def feat_log_turnover_21(panel: Panel) -> pd.DataFrame:
    # Trailing 21-TRADING-day mean of log(turnover).  Single-day turnover is too
    # noisy.  log() drops zero/masked turnover (already NaN in v2/v3) before the
    # roll.  Rolled over trading rows only (see _trailing_mean_over_trading_rows).
    lt = np.log(panel.turnover.where(panel.turnover > 0))
    return _trailing_mean_over_trading_rows(lt, win=21, min_obs=21)


def feat_overnight_tilt_21(panel: Panel) -> pd.DataFrame:
    # Trailing 21-TRADING-day mean overnight return (Lou-Polk-Skouras split).
    return _trailing_mean_over_trading_rows(panel.overnight_ret, win=21, min_obs=21)


def feat_amihud_lagged(panel: Panel) -> pd.DataFrame:
    """Amihud illiquidity, explicitly lagged by one month (fact #19).

    Rebalance date t falls in month M.  We attach Amihud of month M-1, which is
    fully realized before month M begins -- strictly no within-month look-ahead.
    Returned as a wide (date x ticker) frame carrying, for every daily row, the
    value that WOULD be attached if that date were a rebalance date.  Only the
    rebalance rows are ever read downstream.
    """
    am = panel.amihud_monthly.copy()
    # Drop thin-coverage edge months (default conservative; memory flags these).
    if COL_AMIHUD_EDGE in am.columns:
        n_before = len(am)
        am = am[am[COL_AMIHUD_EDGE] != True].copy()
        feat_amihud_lagged._dropped_edge = n_before - len(am)  # driver can report
    # Parse 'YYYY-MM' string -> month-start Timestamp, single ns unit (fact #27).
    am[COL_AMIHUD_MONTH] = (pd.to_datetime(am[COL_AMIHUD_MONTH], format="%Y-%m")
                            .values.astype("datetime64[M]").astype("datetime64[ns]"))
    # Shift each month's value forward to the NEXT month (M-1 -> attached in M).
    am = am.sort_values([COL_AMIHUD_TICKER, COL_AMIHUD_MONTH])
    am["attach_month"] = am[COL_AMIHUD_MONTH] + pd.offsets.MonthBegin(1)
    lookup = am.set_index([COL_AMIHUD_TICKER, "attach_month"])[COL_AMIHUD_VALUE]

    out = pd.DataFrame(index=panel.dates, columns=panel.tickers, dtype=float)
    month_of_row = pd.DatetimeIndex(panel.dates).values.astype("datetime64[M]").astype("datetime64[ns]")
    row_month = pd.Series(month_of_row, index=panel.dates)
    for tkr in panel.tickers:
        try:
            s = lookup.loc[tkr]
        except KeyError:
            continue
        out[tkr] = row_month.map(s)
    return out


FEATURES: dict[str, Callable[[Panel], pd.DataFrame]] = {
    "momentum_12_1": feat_momentum_12_1,
    "reversal_1m": feat_reversal_1m,
    "rvol_60": feat_rvol60,
    "log_turnover_21": feat_log_turnover_21,
    "overnight_tilt_21": feat_overnight_tilt_21,
    "amihud_illiq": feat_amihud_lagged,
}

# Higher illiquidity / higher vol are "bad" directionally but MI is sign-blind;
# we keep raw feature values, MI does not care about sign.


# ---------------------------------------------------------------------------
# FORWARD RETURN
# ---------------------------------------------------------------------------

def forward_return(panel: Panel, h: int) -> pd.DataFrame:
    """h-trading-day forward return.  value@t = close[t+h]/close[t]-1 (FUTURE).
    This is the target; it is the ONLY frame that reaches into the future."""
    c = panel.close
    return c.shift(-h) / c - 1.0


# ---------------------------------------------------------------------------
# PER-DATE CROSS-SECTION EXTRACTION  (+ PIT eligibility)
# ---------------------------------------------------------------------------

@dataclass
class DateCrossSection:
    date: pd.Timestamp
    feat_rank: np.ndarray   # (n,) in (0,1)
    fwd_rank: np.ndarray    # (n,) in (0,1)
    n: int


def _rank01(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank -> (rank-0.5)/n in (0,1).  Ties averaged."""
    s = pd.Series(x)
    r = s.rank(method="average").to_numpy()
    return (r - 0.5) / len(r)


def cross_section(panel: Panel, feat_wide: pd.DataFrame, fwd_wide: pd.DataFrame,
                  t: pd.Timestamp, h: int,
                  min_stocks: int = MIN_STOCKS_PER_DATE) -> DateCrossSection | None:
    """Build one date's eligible, rank-transformed cross-section.

    Eligibility (ALL required):
      * in_nifty500_flag == 1   (fact #17: explicit ==1, never !=1)
      * feature value present    (no fill)
      * forward return present   (no fill)
      * forward window does not cross TRAIN_END (no hold-out contamination)
    """
    # Hold-out guard: the date h positions ahead must exist and be <= TRAIN_END.
    i = panel.dates.get_indexer([t])[0]
    if i < 0:
        raise KeyError(f"rebalance date {t} not in panel index.")
    if i + h >= len(panel.dates):
        return None
    fwd_date = panel.dates[i + h]
    if fwd_date > TRAIN_END:
        return None  # would read hold-out; drop this rebalance for this horizon

    flag_row = panel.flag.loc[t]
    feat_row = feat_wide.loc[t]
    fwd_row = fwd_wide.loc[t]

    eligible = (flag_row == 1) & feat_row.notna() & fwd_row.notna()
    n = int(eligible.sum())
    if n < min_stocks:
        return None

    f = feat_row[eligible].to_numpy(dtype=float)
    y = fwd_row[eligible].to_numpy(dtype=float)
    return DateCrossSection(date=t, feat_rank=_rank01(f), fwd_rank=_rank01(y), n=n)


# ---------------------------------------------------------------------------
# MI CORE
# ---------------------------------------------------------------------------

def _ksg_mi(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """KSG MI between two 1-D continuous variables, in nats.  Non-negative by
    construction of the estimator's clipping; sklearn adds tiny noise internally
    to break rank ties, but we also pass random_state for reproducibility."""
    seed = int(rng.integers(0, 2**31 - 1))
    mi = mutual_info_regression(
        x.reshape(-1, 1), y, n_neighbors=KNN_K, random_state=seed, discrete_features=False
    )[0]
    return float(mi)


def per_date_mi_excess(css: list[DateCrossSection], rng: np.random.Generator,
                       n_shuffle: int = N_SHUFFLE) -> np.ndarray:
    """MI_excess computed SEPARATELY for each date's own cross-section.

    For each date:  raw = KSG MI(feature_rank, fwd_rank) on that date's stocks;
    null = mean over `n_shuffle` within-date permutations of the fwd ranks;
    MI_excess(date) = raw - null.

    WHY PER-DATE, NOT POOLED-THEN-RESAMPLED:  a block bootstrap that resamples
    whole dates WITH REPLACEMENT and re-pools the rows feeds DUPLICATE points
    into one KSG call.  Identical points collapse k-NN distances to zero and
    inflate the MI estimate wildly (measured 3-4x on synthetic) -- a spurious
    UPWARD bias.  Estimating one clean MI_excess per date and then bootstrapping
    those SCALARS avoids duplicate-point contamination entirely, and treats the
    date as the independent unit (correct given overlapping monthly forward
    windows).  Returns one MI_excess value per date.
    """
    out = np.empty(len(css))
    for j, c in enumerate(css):
        raw = _ksg_mi(c.feat_rank, c.fwd_rank, rng)
        nulls = np.empty(n_shuffle)
        for s in range(n_shuffle):
            nulls[s] = _ksg_mi(c.feat_rank, c.fwd_rank[rng.permutation(c.n)], rng)
        out[j] = raw - nulls.mean()
    return out


def summarise(per_date_excess: np.ndarray, rng: np.random.Generator,
              n_boot: int = N_BOOT) -> dict:
    """Point estimate = mean of per-date MI_excess.  CI = bootstrap over the
    per-date scalars (each date one independent draw)."""
    if len(per_date_excess) < 3:
        return {"mi_excess": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "sig": False, "n_dates": len(per_date_excess)}
    m = len(per_date_excess)
    boot = np.array([per_date_excess[rng.integers(0, m, m)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, CI_LOW), np.percentile(boot, CI_HIGH)
    return {"mi_excess": float(per_date_excess.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "sig": bool(lo > 0), "n_dates": m}


# ---------------------------------------------------------------------------
# DRIVER: FULL-SAMPLE HEADLINE + LOOKBACK STATIONARITY PROBE
# ---------------------------------------------------------------------------

def _collect_cross_sections(panel, feat_wide, fwd_wide, rebal_dates, h):
    css, dropped = [], []
    for t in rebal_dates:
        cs = cross_section(panel, feat_wide, fwd_wide, t, h)
        if cs is None:
            dropped.append(t)
        else:
            css.append(cs)
    return css, dropped


def run_screen(panel: Panel,
               features: dict[str, Callable] = None,
               horizons: dict[str, int] = None,
               lookbacks_years: list[int] = None,
               rng_seed: int = RNG_SEED,
               verbose: bool = True) -> dict:
    """Full screen.  Returns a nested result dict; does not print tables (the
    caller / report layer formats).  Each (feature,horizon) yields:
      - headline MI_excess over ALL valid rebalance dates + block-bootstrap CI
      - coverage: n_dates, mean stocks/date
      - lookback probe: for each L, MI_excess over stepped trailing-L windows
        (mean, min, max, n_windows) -- the stationarity view.
    """
    features = features or FEATURES
    horizons = horizons or HORIZONS
    lookbacks_years = lookbacks_years or LOOKBACKS_YEARS
    rng = np.random.default_rng(rng_seed)

    rebal_all = month_end_rebalance_dates(panel)
    # Precompute forward-return frames once per horizon.
    fwd_frames = {name: forward_return(panel, h) for name, h in horizons.items()}
    # Precompute feature frames once.
    feat_frames = {fn: fnc(panel) for fn, fnc in features.items()}

    results = {}
    for fname, feat_wide in feat_frames.items():
        results[fname] = {}
        for hname, h in horizons.items():
            fwd_wide = fwd_frames[hname]
            css, dropped = _collect_cross_sections(panel, feat_wide, fwd_wide, rebal_all, h)

            if len(css) < 3:
                results[fname][hname] = {
                    "n_dates": len(css), "note": "too few valid dates", }
                if verbose:
                    print(f"  {fname:>18} | {hname:>3}: only {len(css)} dates -> skip")
                continue

            # One clean MI_excess per date, then summarise + bootstrap the scalars.
            pde = per_date_mi_excess(css, rng)
            dates_arr = pd.DatetimeIndex([c.date for c in css])
            summ = summarise(pde, rng)
            mean_n = float(np.mean([c.n for c in css]))

            # Lookback stationarity probe: mean per-date excess within trailing-L
            # windows stepped every 12 months.  Uses the SAME per-date scalars, so
            # it is just a windowed average -- cheap and coherent with the headline.
            lb = {}
            for L in lookbacks_years:
                span = pd.Timedelta(days=int(round(365.25 * L)))
                ends = pd.date_range(dates_arr.min() + span, dates_arr.max(), freq="12MS")
                if len(ends) == 0:
                    ends = pd.DatetimeIndex([dates_arr.max()])
                win_ex = []
                for e in ends:
                    sel = (dates_arr > e - span) & (dates_arr <= e)
                    if sel.sum() >= 3:
                        win_ex.append(float(pde[sel].mean()))
                lb[f"{L}y"] = {
                    "mean": float(np.mean(win_ex)) if win_ex else np.nan,
                    "min": float(np.min(win_ex)) if win_ex else np.nan,
                    "max": float(np.max(win_ex)) if win_ex else np.nan,
                    "n_windows": len(win_ex),
                }

            results[fname][hname] = {
                "n_dates": summ["n_dates"],
                "n_dropped": len(dropped),
                "mean_stocks_per_date": mean_n,
                "mi_excess": summ["mi_excess"],
                "ci_low": summ["ci_low"],
                "ci_high": summ["ci_high"],
                "sig": summ["sig"],            # CI excludes zero from below
                "lookback": lb,
            }
            if verbose:
                star = "*" if summ["sig"] else " "
                print(f"  {fname:>18} | {hname:>3}: "
                      f"MI_excess={summ['mi_excess']:+.4f} "
                      f"CI[{summ['ci_low']:+.4f},{summ['ci_high']:+.4f}] {star} "
                      f"(n_dates={summ['n_dates']}, ~{mean_n:.0f} stk/date)")
    return results


# ---------------------------------------------------------------------------
# FEATURE x FEATURE REDUNDANCY MATRIX  (honesty about non-independence)
# ---------------------------------------------------------------------------

def feature_redundancy_matrix(panel: Panel, features: dict = None,
                              rng_seed: int = RNG_SEED) -> pd.DataFrame:
    """MI_excess between each pair of features, pooled over the monthly grid with
    the same within-date-shuffle null.  Shows how much the 'independent' feature
    set actually overlaps.  Symmetric; diagonal set to NaN."""
    features = features or FEATURES
    rng = np.random.default_rng(rng_seed)
    rebal = month_end_rebalance_dates(panel)
    frames = {fn: fnc(panel) for fn, fnc in features.items()}
    names = list(features.keys())
    M = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            fa, fb = frames[names[a]], frames[names[b]]
            css = []
            for t in rebal:
                flag_row = panel.flag.loc[t]
                ra, rb = fa.loc[t], fb.loc[t]
                elig = (flag_row == 1) & ra.notna() & rb.notna()
                if int(elig.sum()) < MIN_STOCKS_PER_DATE:
                    continue
                css.append(DateCrossSection(
                    date=t,
                    feat_rank=_rank01(ra[elig].to_numpy(float)),
                    fwd_rank=_rank01(rb[elig].to_numpy(float)),
                    n=int(elig.sum()),
                ))
            if len(css) >= 3:
                pde = per_date_mi_excess(css, rng)
                ex = float(pde.mean())
                # A high MI_excess between two features is NOT automatically an
                # artifact. Two features built from a shared quantity (e.g. Amihud
                # = mean|ret|/turnover, whose DENOMINATOR is turnover) carry
                # genuine mutual information that rank-correlation misses because
                # the relationship is non-monotone. That entanglement is exactly
                # what this matrix exists to expose -- suppressing it would hide a
                # real finding. So we only NaN out TRUE degeneracy (near-identical
                # ranks -> KSG k-NN collapse), and otherwise REPORT the value,
                # annotating when it is high so the reader interrogates the
                # construction rather than reading it as an independent overlap.
                degen = np.mean([
                    abs(np.corrcoef(c.feat_rank, c.fwd_rank)[0, 1]) > 0.999
                    for c in css
                ])
                if degen > 0.05:
                    warnings.warn(
                        f"[redundancy] {names[a]}<->{names[b]}: {degen:.1%} of dates "
                        f"have |rank-corr|>0.999 (degenerate cross-section, KSG "
                        f"unreliable). Reporting NaN."
                    )
                    M.iloc[a, b] = np.nan
                    M.iloc[b, a] = np.nan
                    M.attrs.setdefault("degenerate_pairs", []).append(
                        (names[a], names[b], ex, float(degen)))
                else:
                    M.iloc[a, b] = ex
                    M.iloc[b, a] = ex
                    if ex > MI_REDUNDANCY_HIGH:
                        # genuine, but flag for construction-overlap interpretation
                        M.attrs.setdefault("high_mi_pairs", []).append(
                            (names[a], names[b], ex))
    return M


# ===========================================================================
# DIAGNOSTIC EXTENSIONS  (2026-08-10 follow-up batch)
# ---------------------------------------------------------------------------
# All read-only characterization. None selects anything, none touches hold-out.
# ===========================================================================

def feature_persistence(panel: Panel, features: dict = None,
                        horizons_months: list[int] = (1, 3, 6, 12)) -> pd.DataFrame:
    """How persistent is each feature's cross-sectional RANK over time?

    For each feature and each lookahead h (in MONTHS), computes the mean
    Spearman rank correlation between the feature's cross-section on rebalance
    date t and the SAME feature's cross-section h months later, across all t.

    WHY THIS EXISTS: the MI screen showed roughly FLAT MI_excess across forward
    horizons for the strong features. Two explanations: (A) genuine multi-horizon
    signal, or (B) a slow-moving CHARACTERISTIC re-detected at every horizon. If a
    feature is highly persistent (its rank barely changes over 12 months) and its
    MI-flatness tracks that persistence, (B) is the likely story. This measures the
    persistence half of that comparison. NOT a pass/fail test -- a diagnostic whose
    pattern we compare against the MI-flatness pattern.

    Purely a feature-vs-its-own-future-self correlation -- NOT predictive of
    returns, so no leakage concern (we are characterising the feature, not trading).
    """
    features = features or FEATURES
    rebal = month_end_rebalance_dates(panel)
    frames = {fn: fnc(panel) for fn, fnc in features.items()}
    # Map each rebalance date to the rebalance date ~h months ahead.
    rebal_list = list(rebal)
    rows = []
    for fname, fw in frames.items():
        for h in horizons_months:
            corrs = []
            for i, t in enumerate(rebal_list):
                j = i + h  # h rebalance steps ahead == h months (monthly grid)
                if j >= len(rebal_list):
                    continue
                t2 = rebal_list[j]
                a = fw.loc[t]
                b = fw.loc[t2]
                # common eligible names: flagged in on BOTH dates + feature present
                elig = ((panel.flag.loc[t] == 1) & (panel.flag.loc[t2] == 1)
                        & a.notna() & b.notna())
                if int(elig.sum()) < MIN_STOCKS_PER_DATE:
                    continue
                rho = pd.Series(a[elig].values).corr(
                    pd.Series(b[elig].values), method="spearman")
                if np.isfinite(rho):
                    corrs.append(rho)
            rows.append(dict(feature=fname, horizon_months=h,
                             mean_rank_autocorr=float(np.mean(corrs)) if corrs else np.nan,
                             n_pairs=len(corrs)))
    return pd.DataFrame(rows)


def sensitivity_cells(panel: Panel, cells: list[tuple], rng_seed: int = RNG_SEED,
                      shuffle_variants=(15, 30), k_variants=(3, 5)) -> pd.DataFrame:
    """Robustness of selected (feature, horizon) cells to estimator settings.

    cells: list of (feature_name, horizon_name). For each, recompute MI_excess at
    each N_SHUFFLE variant (holding k=KNN_K) and each k variant (holding
    N_SHUFFLE=base). Absolute MI may shift with k -- that is expected and fine;
    what must be STABLE is (a) MI_excess vs shuffle count (noise floor converged)
    and (b) the ORDERING of features under different k (ranking not an artifact).
    """
    global KNN_K
    base_k = KNN_K
    rows = []
    fwd_cache = {}
    feat_cache = {}
    for fname, hname in cells:
        if fname not in feat_cache:
            feat_cache[fname] = FEATURES[fname](panel)
        if hname not in fwd_cache:
            fwd_cache[hname] = forward_return(panel, HORIZONS[hname])
        feat_wide = feat_cache[fname]
        fwd_wide = fwd_cache[hname]
        h = HORIZONS[hname]
        css, _ = _collect_cross_sections(panel, feat_wide, fwd_wide,
                                         month_end_rebalance_dates(panel), h)
        if len(css) < 3:
            continue
        # shuffle-count sensitivity (k fixed at base)
        for nsh in shuffle_variants:
            rng = np.random.default_rng(rng_seed)
            pde = per_date_mi_excess(css, rng, n_shuffle=nsh)
            rows.append(dict(feature=fname, horizon=hname, setting=f"shuffle={nsh}",
                             knn_k=base_k, mi_excess=float(pde.mean()), n_dates=len(css)))
        # k sensitivity (shuffle fixed at base N_SHUFFLE)
        for kk in k_variants:
            KNN_K = kk
            rng = np.random.default_rng(rng_seed)
            pde = per_date_mi_excess(css, rng, n_shuffle=N_SHUFFLE)
            rows.append(dict(feature=fname, horizon=hname, setting=f"k={kk}",
                             knn_k=kk, mi_excess=float(pde.mean()), n_dates=len(css)))
        KNN_K = base_k
    return pd.DataFrame(rows)


# ===========================================================================
# CONDITIONAL MUTUAL INFORMATION  (2026-08-10, quant-eye batch, analysis 1)
# ---------------------------------------------------------------------------
# Question: how much does feature A tell us about forward returns OVER AND ABOVE
# what feature B already tells us?  Non-parametric, via joint-minus-marginal:
#
#   condMI(A | B) = MI( [A,B] ; return ) - MI( B ; return )
#
# The joint term uses a 2-D KSG input, which is noisier and MORE positively
# biased than 1-D.  We handle that bias the SAME way as the base screen: under a
# within-date shuffle of returns, the TRUE conditional MI is zero, so whatever
# the estimator reports under shuffle is the bias floor.  We subtract it:
#
#   condMI_excess(A|B) = raw_condMI - mean(raw_condMI under shuffled returns)
#
# A near-zero excess means A is REDUNDANT given B (adds nothing).  A clearly
# positive excess means A carries information B does not.  Estimation is noisier
# than the base screen (2-D), so CIs are wider by design -- an honest cost of
# staying fully non-parametric rather than assuming linearity.
# ===========================================================================

# NOTE: Conditional MI cannot be built from sklearn.mutual_info_regression alone
# (it returns per-column MI for 2-D input, not joint MI). See conditional_mi.py
# for the dedicated estimator. Left out of this module deliberately.

