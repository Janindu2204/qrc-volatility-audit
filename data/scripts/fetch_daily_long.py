"""Fetch the FULL available DAILY history from Yahoo (unofficial chart API) for the 11 commodity futures used in the daily panel.

Writes data/daily_long/<TICKER>_1d.csv plus data/daily_long/pull_record.csv (pull date, rows, date range). Same endpoint and parsing as
fetch_daily_commodities.py (which keeps the 10-year files untouched). Unadjusted front-month continuous series: roll jumps are included.
Then prints a quality report only (nulls, zero-range bars, jumps > 8% log, overlap with the 10-year file); NO cleaning is applied.
Raw provider data: do not redistribute (Yahoo terms unclear); commit this script, not the CSVs."""
import sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
ROOT = Path(__file__).resolve().parents[2] / "data"
OUT = ROOT / "daily_long"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=0&period2={p2}&interval=1d"
DEFAULT = ["HG=F", "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HO=F", "ZC=F", "ZS=F", "ZW=F"]


def fetch(t):
    p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
    r = requests.get(URL.format(t=t, p2=p2), headers=HEADERS, timeout=60); r.raise_for_status()
    p = r.json()["chart"]["result"][0]; q = p["indicators"]["quote"][0]
    df = pd.DataFrame({"date": pd.to_datetime(p["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date,
                       "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"]})
    n_raw = len(df)
    return df.drop_duplicates("date").reset_index(drop=True), n_raw


def quality(t, df, old):
    ok = df.dropna(subset=["close"]); lr = np.log(ok.close).diff()
    row = dict(ticker=t, rows=len(df), rows_null_close=int(df.close.isna().sum()), null_open=int(df.open.isna().sum()),
               zero_range=int((ok.high == ok.low).sum()), nonpos_close=int((ok.close <= 0).sum()), jumps_gt8pct=int((lr.abs() > 0.08).sum()),
               first=str(ok.date.min()), last=str(ok.date.max()))
    if old is not None:
        m = ok.merge(old, on="date")
        row["overlap_rows"] = len(m); row["overlap_max_abs_close_diff"] = float((m.close - m.close_old).abs().max()) if len(m) else np.nan
    return row


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for t in sys.argv[1:] or DEFAULT:
        try:
            df, n_raw = fetch(t)
        except Exception as e:
            print(t, "FAILED", e); continue
        name = t.replace("=", "_").replace("^", "_")
        df.to_csv(OUT / f"{name}_1d.csv", index=False)
        oldf = ROOT / "daily_commod" / f"{name}_1d.csv"
        old = pd.read_csv(oldf, parse_dates=["date"]).assign(date=lambda d: d.date.dt.date)[["date", "close"]].rename(columns={"close": "close_old"}) if oldf.exists() else None
        r = quality(t, df, old); r["pull_date"] = str(date.today()); rows.append(r)
    rec = pd.DataFrame(rows); rec.to_csv(OUT / "pull_record.csv", index=False)
    pd.set_option("display.width", 250); print(rec.to_string(index=False))
