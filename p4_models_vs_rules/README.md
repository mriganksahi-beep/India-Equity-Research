# Fitted Models versus Simple Rules: What Logistic Regression Reveals about Cross-Sectional Return Prediction in Indian Equities

[Paper (PDF)](main.pdf)

Logistic classifiers on five within-date standardised characteristics (1.8M stock-dates, 2011–2026) predict four within-date rank events over the next 20 trading days, validated on twelve rolling folds (3-year train, 1-year test, 21-day purge).

**Findings**
- Beating the median is barely predictable out of sample (AUC 0.537); being among the most volatile is much more predictable (AUC 0.758), largely because volatility persists.
- The fitted "avoid" portfolios held 96–97% of the same stocks as simply dropping the most volatile fifth; their Sharpe advantage did not survive a multiple-comparison correction.
- The fitted "outperform" model trailed a fixed momentum-minus-volatility rule; that rule was designed after the results were seen, so it is an explanatory comparator, not a prospective winner.

Results are gross of costs and include 2022–2025.

## Contents

| Path | What it is |
|---|---|
| `main.pdf`, `main.tex` | The paper |
| `notebooks/moonshot_ml_sandbox_v1_builder.ipynb` | Builds the stock-date dataset with a leakage wall between features and target |
| `notebooks/moonshot_ml_sandbox_v1_visualization.ipynb` | Feature distributions, correlations, PCA implemented from scratch and checked against scikit-learn |
| `notebooks/moonshot_ml_sandbox_phase2_logistic.ipynb` | Rolling-fold logistic classifiers and out-of-sample evaluation |
| `notebooks/moonshot_ml_sandbox_phase3a_portfolios_v2.ipynb`, `..._phase3b_model_vs_twin.ipynb` | Portfolio tests and model-versus-simple-rule comparisons |
| `notebooks/` (other) | Linear regression, conditional-probability and residualised-target studies that preceded the paper |
| `rerun_phase2_bootstrap.py` | Revision diagnostic: moving-block bootstrap for the classifier intervals and a current-volatility baseline |
| `sources/` | Phase records and build logs |
| `make_figures.py`, `claims.csv` | Figure rebuild and number tracing |

## Rebuilding the figures

```bash
pip install -r ../requirements.txt
python make_figures.py
```

See the repository [README](../README.md) for data licensing, the research protocol and known limitations.
