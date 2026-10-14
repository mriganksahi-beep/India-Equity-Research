# Momentum in the Nifty 500: What Survives in Liquid Names, and Six Enhancement Tests That Did Not Pass Their Decision Rules

[Paper (PDF)](main.pdf)

Equal-weighted 12-1 momentum deciles on the point-in-time Nifty 500, 179 monthly rebalances 2011–2026 (gross of costs; includes the period later reserved as hold-out).

**Findings**
- The top-minus-bottom decile spread averages 1.74% per month; annualised Sharpe 0.85 (95% block-bootstrap interval 0.34–1.50). Deciles are broadly ordered, not strictly monotonic.
- Rebuilding deciles inside the most liquid 30/50/70% of names keeps point estimates positive, but no liquidity floor passed its pre-specified joint screen.
- Six enhancement and diagnostic tests (tax-aware holding, volatility scaling, reversal gating, trend age, liquidity hump, path entropy) were not supported under their own decision rules; three are inconclusive rather than negative.

## Contents

| Path | What it is |
|---|---|
| `main.pdf`, `main.tex` | The paper |
| `notebooks/moonshot_test1_momentum_phaseB_baseline.ipynb` | Baseline decile backtest with block-bootstrap inference |
| `notebooks/moonshot_test1_momentum_liquid_universe.ipynb` | Liquidity-restricted universes (turnover and Amihud) |
| `notebooks/moonshot_test1_momentum_M1_tax.ipynb`, `..._M5_lifecycle.ipynb` | Enhancement tests (tax-aware holding; trend age) |
| `sources/` | Archived decile returns, spreads, sweep results and the findings log for each test |
| `reconcile_months.py` | Reconciles the 179- vs 186-month samples across harnesses |
| `make_figures.py`, `claims.csv` | Figure rebuild and number tracing |

## Rebuilding the figures

```bash
pip install -r ../requirements.txt
python make_figures.py
```

See the repository [README](../README.md) for data licensing, the research protocol and known limitations.
