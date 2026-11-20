"""Build per-asset daily files and the block-eligibility table for long_prereg_multiasset.json.
Cleaning (same spirit as v2): drop rows with a null or non-positive OHLC value or high < low; zero-range days stay in the file (the estimator treats zero variance as missing).
Eligibility of test year Y for an asset: zero-range share <= 10% in the test year AND in the two validation years before it. An asset is INCLUDED if it has >= 8 eligible blocks among 2008-2024."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data/daily_multi"; OUT.mkdir(exist_ok=True)
ASSETS = ["GC", "SI", "PL", "PA", "CL", "NG", "HO", "ZC", "ZS", "ZW"]; rows = []
for a in ASSETS:
    d = pd.read_csv(ROOT / f"data/daily_long/{a}_F_1d.csv", parse_dates=["date"]).set_index("date").dropna(subset=["open", "high", "low", "close"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1) & (d.high >= d.low)]
    d["gap_clean"] = 0.0; d["is_ambiguous"] = False; d.to_csv(OUT / f"{a}.csv")
    z = (d.high == d.low).groupby(d.index.year).mean()
    for y in range(2008, 2025):
        share = max(z.get(y, 1), z.get(y - 1, 1), z.get(y - 2, 1)); rows.append(dict(asset=a, year=y, max_zero_share=round(float(share), 3), eligible=bool(share <= 0.10)))
e = pd.DataFrame(rows); n = e.groupby("asset").eligible.sum(); e["asset_included"] = e.asset.map(n >= 8); e.to_csv(OUT / "eligibility.csv", index=False)
print(n.to_dict()); print("included assets:", sorted(n[n >= 8].index), "| blocks to run:", int(e[e.asset_included & e.eligible].shape[0]))
