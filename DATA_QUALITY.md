# Data quality

Known issues in the point-in-time Nifty 500 panel, how each was found, and how it is handled. The panel used by all papers is `feature_panel_with_pit_v2_2026-07-24` (4,237 trading dates × 1,069 tickers, 2010-01-04 to 2026-07-10).

| Issue | How it was found | Handling | Status |
|---|---|---|---|
| **Placeholder rows from the price vendor.** On days a stock did not trade, Yahoo Finance can return a row with open = high = low = close = 0 and zero volume. | A ±30% daily-return review caught only 0.9% of these rows; the rest were found by searching directly for the all-zero signature. | Rows removed before any feature is computed. | Fixed in v2 panel |
| **Split adjustment.** Unclear whether vendor closes were already split-adjusted. | Checked against 12+ real splits (ratios 1.33×–10×). | Vendor close confirmed split-adjusted; no custom adjustment applied (applying one would double-adjust). | Resolved |
| **Tickers with no usable vendor data.** | Coverage checks per ticker against the longest feature window (absolute valid rows, not NaN share). | 114 tickers sourced from Bloomberg instead; 1,069 of 1,071 validated. Two permanent gaps (MIRCELECTR, DENABANK). | Resolved |
| **Truncated membership snapshot.** The 2018-06-30 index snapshot has ~353 rows instead of ~500 and was carried forward from 2017-12-31. | Row-count check across the 33 semi-annual snapshots. | Up to ~22% of the universe may be misclassified in Jul–Dec 2018; any result concentrated there needs extra scrutiny. | Disclosed |
| **Membership is snapshot-based.** Index membership is reconstructed from semi-annual snapshots into a daily flag (member / not / unknown). | By construction. | Tests use members only (flag = 1); intra-period index changes are approximated. | Disclosed |
| **Vendor failure on 2025-03-18.** | Date-level anomaly checks. | Flagged; no rebalance may fall on it. The one missing return blanks the 60-day realized-volatility feature for ~60 trading days (about Apr–Jun 2025), which removes those months wherever that feature is used. | Disclosed; NaN-tolerant rebuild planned |
| **Exchange-closed dates in the calendar.** The panel's date index contains some NSE holidays on which only a few tickers carry rows. When a month-end or a forward-return endpoint lands on one, that month's cross-section collapses and the month is dropped. | Found during paper review by listing dropped months: most coincide with NSE holidays (e.g. 2014-08-29 Ganesh Chaturthi, 2016-10-31 Diwali Balipratipada, 2020-11-30 Guru Nanak Jayanti). | Disclosed in each paper; a calendar-corrected re-run (coverage audit, then removal of closed or thin dates before any positional shift) is in [`p3_mi_screen/notebooks/moonshot_mi_calfix_rerun.ipynb`](p3_mi_screen/notebooks/moonshot_mi_calfix_rerun.ipynb). | Re-run in progress |
| **Special sessions.** 2015-02-28 was a genuine Saturday trading session (Union Budget day) with thin coverage. | Weekend-date check in the momentum validation flagged it. | Treated as a real session; the calendar audit decides whether its coverage is sufficient for a rebalance. | Disclosed |

## Lessons recorded

- Thresholds catch loud errors; only structural checks catch quiet ones.
- NaN share alone is misleading for ticker usability; check absolute valid rows against the longest feature window.
- Rolling windows over a wide (date × ticker) frame with holiday gaps need to roll over each ticker's trading rows, not calendar rows.
- Every pipeline stage is checkpointed and versioned; derived status files are regenerated from the checkpoint rather than hand-merged.
