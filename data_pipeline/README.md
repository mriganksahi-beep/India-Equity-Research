# Data pipeline: point-in-time Nifty 500 panel

Builds the daily panel used by every paper: prices for every stock that was ever in the Nifty 500 since 2010, a daily index-membership flag reconstructed from semi-annual snapshots, and point-in-time features. Raw data are not included (see the repository README).

| Notebook | Stage | What it does |
|---|---|---|
| `01_pit_data_pilot.ipynb` | 1–2 | Pilot: pulls OHLCV from Yahoo Finance, tests whether vendor closes are already split-adjusted, pulls the full universe with checkpointing, retries failed tickers on the BSE listing, and cross-checks point-in-time membership |
| `02_stage3_production_pipeline.ipynb` | 3 | Production build of the point-in-time universe and raw price panel from two sources (Yahoo Finance, Bloomberg export), with a read-only final validation cell and a diagnosis of tickers with excessive missing closes |
| `03_stage3b_remediation.ipynb` | 3B | Fixes three data-quality defects in the v1 price panel and writes a new versioned v2 panel; v1 is left untouched |
| `04_stage4_6_pipeline_v3.ipynb` | 4–6 | Builds the daily point-in-time feature panel and a separate monthly Amihud illiquidity file; idempotent, resolves the newest file version, routes every path through one resolver |
| `05_phaseA_validation_v3.ipynb` | Validation | Coverage, spot-check, leakage and universe checks run against the panel before any hypothesis test; refuses to run against a pre-fix panel |

Design rules followed throughout: checkpointed and re-runnable stages, versioned outputs that are never overwritten, explicit point-in-time (`as_of`) logic, no silent fills, and status files regenerated from the checkpoint rather than edited by hand. Known issues and their handling are in [`../DATA_QUALITY.md`](../DATA_QUALITY.md).
