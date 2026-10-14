# Indian Equity Research

Independent quantitative research on the cross-section of Indian equities (Nifty 500, 2010–2026) by **Mrigank Sahi**, Investment Research Associate, New Delhi.

The project asks a simple question with a strict process: *which publicly available price and volume signals carry real, usable information about future returns of Indian stocks?* Every hypothesis has its decision rule written down before it is tested, every pipeline is checked on synthetic data with known answers before it touches real data, and negative results are reported with the same care as positive ones.

## Papers

| | Paper | Question | Main finding | Folder |
|---|---|---|---|---|
| 1 | **Mutual Information in the Cross-Section of Indian Equities** | What does a non-linear dependence measure add over the standard information coefficient? | All six common signals carry information about future returns, but for more than half of the signal–horizon pairs its *direction* flips from month to month, so an average IC near zero hides a strong monthly relationship. A first conditional-MI analysis showed spurious "synergy"; the cause was traced to an estimator mismatch and corrected. | [`p3_mi_screen`](p3_mi_screen) |
| 2 | **Fitted Models versus Simple Rules** | Do logistic classifiers add value beyond simple characteristic sorts? | Relative volatility is predictable a month ahead (AUC 0.76), but mostly because volatility persists; return direction barely is (AUC 0.54). The fitted "avoid" portfolios held 96–97% of the same stocks as simply dropping the most volatile fifth, and the fitted "outperform" model trailed a simple momentum-minus-volatility rule (a comparator designed after the results were seen). | [`p4_models_vs_rules`](p4_models_vs_rules) |
| 3 | **Path Texture and the Cross-Section of Indian Equity Returns** | Does the *shape* of a stock's recent price path (permutation entropy, variance ratio) predict returns? | No. Both measures failed their first pre-specified gate; the ordinal structure of daily NSE returns is close to that of random noise for almost every stock. | [`p2_path_texture`](p2_path_texture) |
| 4 | **Momentum in the Nifty 500** | Does 12-1 momentum work in a point-in-time Nifty 500, in liquid names, and can six enhancements improve it? | A positive, broadly ordered momentum spread (gross of costs, full sample); point estimates survive in liquid subsets, but no liquidity floor met its pre-specified bar and none of six enhancements passed its decision rule. | [`p1_momentum`](p1_momentum) |

Each folder contains the paper (PDF and LaTeX source), the notebooks and code that produced it, the derived results, a `claims.csv` that traces every number in the paper to its source file, and a script that rebuilds the figures.

**Related work.** *Regime-Dependent Price Formation in Indian Equities: Evidence from Factor Information Coefficients and Portfolio Microstructure* — under review at the *Pacific-Basin Finance Journal* (not included here).

## What this repository demonstrates

- **Data engineering.** A point-in-time Nifty 500 panel built from two price sources and semi-annual index-membership snapshots, with checkpointed, re-runnable pipelines and versioned outputs ([`data_pipeline`](data_pipeline), [`DATA_QUALITY.md`](DATA_QUALITY.md)).
- **Validation discipline.** Synthetic planted-signal self-tests gate every real-data run; leakage, universe and hold-out guards are enforced in code and re-audited; results were reproduced to floating-point precision on re-run.
- **Statistical inference.** Moving-block bootstrap intervals for overlapping horizons, permutation nulls, Holm multiple-testing correction, Newey–West and Fama–MacBeth regressions, estimator calibration against known answers.
- **Machine learning.** Logistic classification with rolling, purged validation folds; out-of-sample skill and AUC; comparison against transparent baselines.
- **Honest reporting.** A running test ledger ([`TEST_LEDGER.md`](TEST_LEDGER.md)), explicit disclosure of what each paper does *not* establish, and limitations stated up front.

## Research protocol

1. Write the hypothesis, the exact signal definition and the pass/fail rule before running anything.
2. Build the test and run it on synthetic data with a planted signal and a planted null; it must find the first and reject the second.
3. Run on real data for the 2010–2021 development period; the later period is reserved.
4. Report the result with intervals, whatever it is, and log it in the ledger.

## Repository layout

```
data_pipeline/        point-in-time panel construction (Stages 1–6) and validation notebooks
p1_momentum/          momentum paper:            main.pdf, notebooks/, sources/, make_figures.py
p2_path_texture/      path-texture paper
p3_mi_screen/         mutual-information paper:  code/ (Python modules + self-tests), notebooks/
p4_models_vs_rules/   models-vs-rules paper
common/               shared LaTeX preamble and bibliography
TEST_LEDGER.md        every test run, its family, outcome and where it is reported
DATA_QUALITY.md       known data issues and how they were handled
```

## Data and licensing

Raw prices (Yahoo Finance) and index-membership snapshots (licensed vendor) are **not redistributed**. The pipeline notebooks document how the panel was built; the folders here contain code, derived statistics (per-date or per-cell results, not prices) and the papers. Code is released under the MIT licence; the papers remain © the author.

## Known limitations

- Several analyses (momentum, the original entropy test, the ML study) used data from 2022 onward, so that period is not an untouched hold-out for those features. Each paper says which period it used.
- The panel's calendar contained exchange-closed and thin-coverage dates, which silently dropped some months from monthly tests. The mutual-information paper has been re-run on an audited calendar (165 dates removed); for momentum the dropped months were not neutral (one-month IC 0.085 before, 0.069 after). The other three papers still use the original calendar and disclose it.
- All portfolio results are gross of transaction costs.

## Use of AI tools

Code was written with the assistance of an AI coding assistant (Anthropic's Claude) to specifications set by the author, and the manuscripts were drafted with the same assistance. The author designed the research questions and decision rules, ran the analyses and reviewed all code and text.

## Running the code

```bash
pip install -r requirements.txt
cd p3_mi_screen && python make_figures.py        # rebuilds figures from the archived results
```

The notebooks were written for Google Colab with data on Google Drive; paths at the top of each notebook point to that layout.
