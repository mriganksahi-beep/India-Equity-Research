"""Revision diagnostic (post hoc, added in round 2): reconcile Phase B's 179 monthly
observations with the M1/M3 harness's 186, using only archived artifacts.

Evidence used
  sources/long_short_spread_2026-07-27.csv  Phase B rebalance dates (179)
  Phase B notebook Cell B1 printout          the 19 skipped month-ends (listed below)
  sources/m1_holdout_2026-07-30.csv          M1 baseline n_train=131 (<2022-01-01), n_holdout=55
  M1 notebook code                            months are indexed by the window END date
                                              (rows.append({"rb": m_end, ...})) and every
                                              month-end pair is marked; when ranking fails
                                              (<100 names) the prior book is kept.
"""
import pandas as pd

pb = pd.read_csv("sources/long_short_spread_2026-07-27.csv", parse_dates=["rebalance_date"])
m1 = pd.read_csv("sources/m1_holdout_2026-07-30.csv").set_index("variant").loc["baseline"]

skipped_mid = pd.to_datetime(["2011-08-31", "2014-08-29", "2015-02-28", "2016-10-31",
                              "2018-03-29", "2019-03-29", "2020-11-30"])
start, end = pb.rebalance_date.min(), pb.rebalance_date.max()
# Candidate monthly windows between the first and last Phase B rebalance (inclusive):
n_months_calendar = (end.year - start.year) * 12 + (end.month - start.month) + 1
assert not set(skipped_mid) & set(pb.rebalance_date)
assert n_months_calendar - len(skipped_mid) == len(pb) == 179
print(f"Phase B: {start.date()}..{end.date()}: {n_months_calendar} month-start windows,"
      f" {len(skipped_mid)} skipped mid-sample -> {len(pb)}")

# M1 indexes a window by its END; train = window end < 2022-01-01.
# Window ends 2011-02 .. 2021-12 = 131 months; 2022-01 .. 2026-07 = 55 months.
train_calendar = (2021 - 2011) * 12 + (12 - 2) + 1
hold_calendar = (2026 - 2022) * 12 + 7
assert (int(m1.n_train), int(m1.n_holdout)) == (train_calendar, hold_calendar) == (131, 55)
print(f"M1: n_train={int(m1.n_train)}, n_holdout={int(m1.n_holdout)} = every calendar month,"
      " i.e. the same first and last window as Phase B with no interior gaps")

# Phase B windows split the same way (window start <= 2021-11-30 <-> end <= 2021-12-31):
pb_train = (pb.rebalance_date <= "2021-11-30").sum()
pb_hold = (pb.rebalance_date >= "2021-12-31").sum()
print(f"Phase B same split: {pb_train} / {pb_hold}  (differences: {131-pb_train} / {55-pb_hold})")
assert (131 - pb_train, 55 - pb_hold) == (7, 0)
print("=> all 7 extra M1 months are the 7 interior months Phase B skipped; endpoints identical.")
