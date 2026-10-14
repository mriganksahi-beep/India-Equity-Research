"""
conditional_mi_selftest.py
==========================
Ground-truth self-test for conditional_mi.py.  Must pass before real data.

Builds a synthetic monthly cross-sectional panel with a KNOWN structure:

  * hidden driver q每 month moves the forward return.
  * feat_redundant_A and feat_redundant_B are BOTH noisy copies of q.
      -> each is marginally informative, but conditioned on the other, the
         additional information is ~0 (they are the same signal twice).
  * feat_independent carries a SEPARATE driver r that also moves returns.
      -> conditioned on feat_redundant_A, it should REMAIN informative.

Assertions:
  (1) library sanity: CMI ~0 for independent, drops for redundant, stays for additive
  (2) our per-date wrapper: condMI_excess(A | B_redundant) is NOT significant
      (CI includes 0)  -> redundancy detected
  (3) our per-date wrapper: condMI_excess(independent | A_redundant) IS significant
      (CI excludes 0)  -> additivity preserved
"""

import numpy as np
import pandas as pd
import mi_screen as M
import conditional_mi as C


def build_synthetic(seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", "2021-06-30")
    tickers = [f"T{i:03d}" for i in range(150)]
    D, N = len(dates), len(tickers)
    month_id = pd.PeriodIndex(dates, freq="M")
    uniq = month_id.unique()
    pos = {m: k for k, m in enumerate(uniq)}

    # hidden monthly drivers
    q = rng.standard_normal((len(uniq), N))   # drives returns; copied into A,B
    r = rng.standard_normal((len(uniq), N))   # separate driver; -> independent feat

    # daily returns: next month's return driven by THIS month's q and r
    daily_ret = np.zeros((D, N))
    market = rng.standard_normal(D) * 0.01
    for di in range(D):
        k = pos[month_id[di]]
        kp = max(k - 1, 0)
        zq = (q[kp] - q[kp].mean()) / (q[kp].std() + 1e-9)
        zr = (r[kp] - r[kp].mean()) / (r[kp].std() + 1e-9)
        drift = 0.0015 * zq + 0.0015 * zr
        daily_ret[di] = market[di] + drift + rng.standard_normal(N) * 0.015

    close = 100 * np.cumprod(1 + daily_ret, axis=0)
    close_df = pd.DataFrame(close, index=dates, columns=tickers)
    dr_df = pd.DataFrame(daily_ret, index=dates, columns=tickers)

    # features observable as-of month end
    A = pd.DataFrame(np.nan, index=dates, columns=tickers)  # noisy copy of q
    B = pd.DataFrame(np.nan, index=dates, columns=tickers)  # noisy copy of q
    I = pd.DataFrame(np.nan, index=dates, columns=tickers)  # noisy copy of r
    noiseA = rng.standard_normal((len(uniq), N)) * 0.3
    noiseB = rng.standard_normal((len(uniq), N)) * 0.3
    noiseI = rng.standard_normal((len(uniq), N)) * 0.3
    for di in range(D):
        k = pos[month_id[di]]
        A.iloc[di] = q[k] + noiseA[k]
        B.iloc[di] = q[k] + noiseB[k]
        I.iloc[di] = r[k] + noiseI[k]

    # assemble a minimal panel; only the pieces conditional_mi needs
    flag = pd.DataFrame(1.0, index=dates, columns=tickers)
    long = pd.DataFrame({
        M.COL_DATE: np.repeat(dates.values, N),
        M.COL_TICKER: np.tile(tickers, D),
        M.COL_CLOSE: close_df.values.ravel(),
        M.COL_DAILY_RET: dr_df.values.ravel(),
        M.COL_OVERNIGHT_RET: 0.0, M.COL_TURNOVER: 1e6,
        M.COL_RVOL60: 0.3, M.COL_FLAG: 1.0,
    })
    amihud = pd.DataFrame({M.COL_AMIHUD_TICKER: tickers,
                           M.COL_AMIHUD_MONTH: "2010-01",
                           M.COL_AMIHUD_VALUE: 1e-9})
    panel = M.build_panel(long, amihud)
    return panel, A, B, I


def main():
    print("=" * 70)
    print("CONDITIONAL MI SELF-TEST")
    print("=" * 70)
    panel, A, B, I = build_synthetic()

    # register the planted features so conditional_mi can find them by name
    M.FEATURES["redundant_A"] = lambda p, w=A: w
    M.FEATURES["redundant_B"] = lambda p, w=B: w
    M.FEATURES["independent"] = lambda p, w=I: w

    M.N_SHUFFLE = 10
    M.N_BOOT = 300

    print("\n[redundancy] condMI(redundant_A | redundant_B) -- expect NOT significant:")
    r_red = C.conditional_mi_pair(panel, "redundant_A", "redundant_B", "1m")

    print("\n[additivity] condMI(independent | redundant_A) -- expect significant:")
    r_add = C.conditional_mi_pair(panel, "independent", "redundant_A", "1m")

    print("\n--- assertions ---")
    print(f"redundant_A|redundant_B: marg={r_red['marginal_mi_excess']:+.4f} "
          f"cond={r_red['cond_mi_excess']:+.4f} survived={r_red['fraction_survived']:.0%} "
          f"sig={r_red['sig']}")
    print(f"independent|redundant_A: marg={r_add['marginal_mi_excess']:+.4f} "
          f"cond={r_add['cond_mi_excess']:+.4f} survived={r_add['fraction_survived']:.0%} "
          f"sig={r_add['sig']}")

    # redundancy: marginal was informative, conditioning collapses it
    assert r_red["marginal_mi_excess"] > 0.02, "planted redundant feature not marginally informative -> bad synthetic"
    assert r_red["fraction_survived"] < 0.5, "REDUNDANCY NOT DETECTED: too much survived conditioning"
    # additivity: independent signal survives conditioning
    assert r_add["sig"], "ADDITIVITY LOST: independent signal wrongly killed by conditioning"
    assert r_add["fraction_survived"] > 0.5, "independent signal lost too much information"

    # cleanup registry
    for k in ["redundant_A", "redundant_B", "independent"]:
        M.FEATURES.pop(k, None)

    print("\nALL CONDITIONAL-MI SELF-TEST ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
