# Data (not included; rebuild with the scripts below)

| Data | Source | How to get it | Lands in |
|---|---|---|---|
| Li et al. S&P 500 realized volatility, predictors, couplings and published forecasts | github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting, commit d2e9b0aa599afd33c6cfb10352e2c36fe92fa76f | `python tests/audit/fetch_li.py` (checks the SHA-256 values in `data/li2026/SOURCE.json`) | `data/li2026/` |
| Daily commodity futures (copper and others), VIX, dollar index | Yahoo Finance chart interface | `python data/scripts/fetch_daily_long.py`, then `python data/scripts/flag_copper_long.py v2` | `data/daily_long/` |
| Per-asset daily panel and eligibility table | built from `data/daily_long/` | `python data/scripts/ma_build.py` | `data/daily_multi/` |

`data/li2026/t8_U1.npy` and `t8_inputs.npz` (used by the hardware test T8) are created by `tests/audit/t8_plan.py`.
Yahoo data change over time; the exact pull dates of the paper are listed in the run logs in `results/logs/`.
