# Mutual Information in the Cross-Section of Indian Equities: Sign Instability, Feature Overlap, and Conditional-MI Calibration

[Paper (PDF)](main.pdf)

Screens six price and volume features (12-1 momentum, 1-month reversal, 60-day realized volatility, 21-day log turnover, 21-day overnight-return tilt, lagged Amihud illiquidity) against forward returns at five horizons, 2010–2021, using a rank-based mutual-information estimator with a within-date permutation null.

**Findings**
- All 30 feature–horizon cells carry information above zero (95% moving-block bootstrap; all survive a Holm correction). Only 13 have a mean rank IC that survives the same correction.
- The gap is mostly *sign instability*: relationships that are strong in a given month but change sign across months. An averaged IC cancels them; sign-blind MI does not.
- Between features, MI is most useful where correlation is weakest: reversal and momentum are both U-shaped in volatility.
- A first conditional-MI analysis showed "synergy" for every pair. Diagnostics traced it to two estimators on different scales and a wrong noise benchmark; the corrected analysis shows net redundancy between momentum and volatility and no synergy.

The development period only is used; every forward window ends by 2021-12-31 (enforced in code and re-audited).

## Contents

| Path | What it is |
|---|---|
| `main.pdf`, `main.tex` | The paper |
| `code/mi_screen.py` | Point-in-time MI screen: panel construction, features, per-date KSG MI with permutation null |
| `code/conditional_mi.py` | Conditional MI, estimator diagnostics, and the corrected noise-floor metric |
| `code/mi_paper_rerun.py` | Reproduction with per-date capture, moving-block bootstrap, direction/extremeness split, KSG calibration |
| `code/*selftest.py` | Synthetic planted-signal tests that gate every real-data run |
| `notebooks/` | Original screen driver (2026-08-10), the reproduction re-run, and the calendar-corrected re-run |
| `sources/`, `derived/` | Archived per-cell and per-date statistics (no prices) |
| `make_figures.py` | Rebuilds every figure and derived number from `sources/` |
| `claims.csv` | Every number in the paper, traced to its source file |

## Rebuilding the figures

```bash
pip install -r ../requirements.txt
python make_figures.py
```

See the repository [README](../README.md) for data licensing, the research protocol and known limitations.
