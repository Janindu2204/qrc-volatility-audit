"""[v2 mode: `python flag_copper_long.py v2` -> flagged_v2/, roll-suspect 2005-05-27 and 2005-11-29, 2004-10-13 only marked (expiry check did not support a roll)]
Flag copper (HG=F) long daily history and blank the two roll-suspect days. Reads data/daily_long/HG_F_1d.csv, writes data/daily_long/flagged/HG_F_1d.csv.

Flags (no rows removed): is_zero_range (high == low), is_jump8 (|log close-to-close| > 8%), is_roll_suspect, is_ambiguous.
ret_clean / gap_clean = close-to-close return and overnight gap with the roll-suspect days set to NaN (prices untouched: the day's
high/low come from one contract; blanking the close would push the roll jump into the next day's return).
Roll-suspect dates come from a manual cross-check (gap ~ whole move, small hi-lo range, gold/silver/crude flat); NOT checked against
actual contract expiry dates. Ambiguous dates are only marked."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "daily_long"
import sys
V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"
ROLL = pd.to_datetime(["2005-05-27", "2005-11-29"] if V2 else ["2004-10-13", "2005-05-27"])
AMBIG = pd.to_datetime(["2004-10-13", "2005-01-04", "2007-01-03", "2009-02-06"] if V2 else ["2005-01-04", "2007-01-03", "2009-02-06"])
SUB = "flagged_v2" if V2 else "flagged"

if __name__ == "__main__":
    d = pd.read_csv(ROOT / "HG_F_1d.csv", parse_dates=["date"]).dropna(subset=["close"]).reset_index(drop=True)
    d["ret"] = np.log(d.close).diff(); d["gap"] = np.log(d.open / d.close.shift(1))
    d["is_zero_range"] = (d.high == d.low); d["is_jump8"] = d.ret.abs() > 0.08
    d["is_roll_suspect"] = d.date.isin(ROLL); d["is_ambiguous"] = d.date.isin(AMBIG)
    bad = d.is_roll_suspect
    d["ret_clean"] = d.ret.mask(bad); d["gap_clean"] = d.gap.mask(bad)
    (ROOT / SUB).mkdir(exist_ok=True); d.to_csv(ROOT / SUB / "HG_F_1d.csv", index=False)
    print(len(d), "rows;", {c: int(d[c].sum()) for c in ["is_zero_range", "is_jump8", "is_roll_suspect", "is_ambiguous"]},
          "; NaN ret_clean:", int(d.ret_clean.isna().sum()), "(1 first row + 2 roll days expected = 3)")
    print(d[d.is_roll_suspect][["date", "ret", "gap", "ret_clean", "gap_clean"]].to_string(index=False))
