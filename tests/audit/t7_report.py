"""Score Test 7 as registered in prereg_li_audit_T7.json. Usage: python t7_report.py > report_T7.txt
Addendum A1 (constant column in RandTrig and RFF): python t7_report.py fix > report_T7_A1.txt"""
import sys
from pathlib import Path
import numpy as np
from arch.bootstrap import SPA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t7_run as T
from stats_lib import compare, table, SEED

P = dict(np.load(HERE / "out" / "preds_T7.npz"))
if sys.argv[1:] == ["fix"]:
    P.update(np.load(HERE / "out" / "preds_ctrlfix.npz"))
    print("ADDENDUM A1 (post-hoc deviation): RandTrig and RFF = p-1 random features + constant column. Registered result: report_T7.txt")
P_OF = {"QR1": 10, "QR2": 20}
edges = {"QR1": [], "QR2": []}

for asset in T.ASSETS:
    M, S, ntr, sc = T.build(asset); y = S.RV.values
    nval = 36; YT, YV = y[ntr:], y[ntr - nval:ntr]
    lo, hi, A, B = sc["RV"]
    unsc = lambda z: lo + (z - A) * (hi - lo) / (B - A)
    LOSS = {"primary": lambda p, t: (p - t) ** 2,
            "secondary": lambda p, t: (lambda hf, ha: ha / hf - np.log(ha / hf) - 1)(np.exp(2 * unsc(p)), np.exp(2 * unsc(t)))}
    for design in ("QR1", "QR2"):
        p = P_OF[design]
        QV = np.array([P[f"{asset}_{design}_{i:03d}_va"] for i in range(100)])
        QT = np.array([P[f"{asset}_{design}_{i:03d}_te"] for i in range(100)])
        iv = int(np.argmin(((QV - YV) ** 2).mean(1))); qv = QT[iv]
        mse = ((QT - YT) ** 2).mean(1)
        har = P[f"{asset}_{design}_HAR_te"]
        print("=" * 100); print(f"{asset} {design}: 100-instance test MSE median {np.median(mse):.5f} [5-95% {np.percentile(mse, 5):.5f}, {np.percentile(mse, 95):.5f}]; "
              f"HAR {((har - YT) ** 2).mean():.5f}; instances beating HAR {int((mse < ((har - YT) ** 2).mean()).sum())}/100; val-selected instance {iv + 1}")
        pick = lambda k: P[f"{asset}_{design}_{k}_te"][int(np.argmin(((P[f'{asset}_{design}_{k}_va'] - YV) ** 2).mean(1)))]
        ctrl = {f"ESNtuned-{p}": pick("ESNtuned"), f"RandTrig-{p}": pick("RandTrig"), f"RFF-{p}": pick("RFF"),
                "Trig2": P[f"{asset}_{design}_Trig2_te"][0], "QR_J0": P[f"{asset}_{design}_J0_te"],
                "AR3": P[f"{asset}_{design}_AR3_te"], "HAR": har}
        for lname, lf in LOSS.items():
            rows = [(f"QR_val vs {nm}", *compare(lf(qv, YT), lf(b, YT))) for nm, b in ctrl.items()]
            ph = table(rows, f"{asset} {design} loss={lname} (Holm over 7)")
            if lname == "primary":
                edge = all(r[1] < 1 and h < 0.05 for r, h in zip(rows, ph)); edges[design].append((asset, edge))
                Mx = np.column_stack([lf(QT[i], YT) for i in range(100)])
                spa = SPA(lf(har, YT), Mx, block_size=6, reps=10000, bootstrap="stationary", seed=SEED); spa.compute()
                print(f"  SPA of 100 instances vs HAR: p = {float(spa.pvalues['consistent']):.4f};  quantum edge (strict rule): {edge}")

print("\nOVERALL (registered: transfer supported only if edge in >= 5 of 8 assets)")
for d, e in edges.items():
    n = sum(x for _, x in e)
    print(f"  {d}: edge in {n}/8 assets {[a for a, x in e if x]} -> {'SUPPORTED' if n >= 5 else 'NOT SUPPORTED on commodities'}")
