"""Score Test 4a (prereg_li_audit_T4T6.json): published full-sample input scaling vs train-window-only scaling.
AR3 and HAR use only the published RV target lags plus an intercept, so they are the same under both scalings.
Usage: python t4a_report.py > report_T4a.txt"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L
from stats_lib import compare, table

O = np.load(HERE / "out" / "preds_T2T3.npz"); N = np.load(HERE / "out" / "preds_T4a.npz")
y = L.load_data().RV.values; YT, YV = y[L.WI:], y[451:L.WI]
LOSS = {"primary": lambda p, t: (p - t) ** 2 * L.DIF ** 2, "secondary": L.qlike_var_loss}
AUTH = {"QR1": 0, "QR2": 1}

for design in ("QR1", "QR2"):
    grab = lambda P, s: np.array([P[f"{design}_{i:03d}_{s}"] for i in range(100)])
    sets = {"published": (grab(O, "va"), grab(O, "te")), "train-only": (grab(N, "va"), grab(N, "te"))}
    print("=" * 100); print(design); print("=" * 100)
    pick = {}
    for nm, (QV, QT) in sets.items():
        mse = ((QT - YT) ** 2).mean(1) * L.DIF ** 2
        iv = int(np.argmin(((QV - YV) ** 2).mean(1))); a = AUTH[design]
        pick[nm] = (QT[a], QT[iv])
        print(f"{nm:10s} scaling: median MSE {np.median(mse):.4f} [5-95% {np.percentile(mse, 5):.4f}, {np.percentile(mse, 95):.4f}]; "
              f"authors' instance {a + 1} MSE {mse[a]:.4f} (rank {int((mse < mse[a]).sum()) + 1}); val-selected instance {iv + 1} MSE {mse[iv]:.4f}")
    ar3, har = O["AR3_te"], O["HAR_te"]
    for lname, lf in LOSS.items():
        rows = [("authors' instance: train-only vs published", *compare(lf(pick["train-only"][0], YT), lf(pick["published"][0], YT))),
                ("val-selected: train-only vs published", *compare(lf(pick["train-only"][1], YT), lf(pick["published"][1], YT)))]
        table(rows, f"{design} loss={lname}: effect of the scaling (ratio < 1 means train-only scaling is better)")
        rows = [(f"{who} (train-only) vs {nm}", *compare(lf(q, YT), lf(b, YT)))
                for who, q in (("authors' instance", pick["train-only"][0]), ("val-selected", pick["train-only"][1]))
                for nm, b in (("AR3", ar3), ("HAR", har))]
        table(rows, f"{design} loss={lname}: train-only scaling vs linear baselines")
