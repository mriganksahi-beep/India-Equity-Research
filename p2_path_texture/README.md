# Path Texture and the Cross-Section of Indian Equity Returns: Permutation Entropy and the Variance Ratio

[Paper (PDF)](main.pdf)

Tests whether the *shape* of a stock's recent daily return path predicts the cross-section of returns, using normalized permutation entropy (order 4) and the 21-day Lo–MacKinlay variance ratio over a 232-row window. Each test had a gate chain written down before it ran.

**Findings**
- Neither measure passed its first gate: Fama–MacBeth Newey–West t-statistics were −0.29 (entropy) and −0.41 (variance ratio) even before controls; all 95% intervals include zero. Later gates were not reached.
- Most stock-months have entropy close to the finite-sample benchmark for random noise; simulated noise of the same length reproduces about two-thirds of the cross-sectional entropy variance.
- The entropy test used 2011–2026 data; a post hoc 2011–2021 re-estimate also fails.

## Contents

| Path | What it is |
|---|---|
| `main.pdf`, `main.tex` | The paper |
| `notebooks/moonshot_test1_momentum_M7_entropy.ipynb` | Permutation-entropy test (gate chain) |
| `notebooks/moonshot_batch2_D_gate.ipynb` | Variance-ratio test |
| `notebooks/moonshot_batch2_C_gate.ipynb` | Related low-volatility gate and its two diagnostics (appendix of the paper) |
| `revision_diagnostics.py` | Revision diagnostics: simulated nulls, tie checks, tail-robust gates (needs raw prices, not included) |
| `sources/entropy_panel.parquet`, `vr_panel.parquet` | Per stock-month entropy and variance-ratio values (derived statistics, no prices) |
| `make_figures.py`, `claims.csv` | Figure rebuild and number tracing |

## Rebuilding the figures

```bash
pip install -r ../requirements.txt
python make_figures.py
```

See the repository [README](../README.md) for data licensing, the research protocol and known limitations.
