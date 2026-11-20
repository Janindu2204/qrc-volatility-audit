"""Score Tests 2-3 exactly as registered in prereg_li_audit_T2T3.json. Usage: python t2t3_report.py > report_T2T3.txt
Addendum A1 (constant column in RandTrig and RFF): python t2t3_report.py fix > report_T2T3_A1.txt"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from arch.bootstrap import StationaryBootstrap, SPA, MCS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L

P = dict(np.load(HERE / "out" / "preds_T2T3.npz"))
if sys.argv[1:] == ["fix"]:
    P.update(np.load(HERE / "out" / "preds_ctrlfix.npz"))
    print("ADDENDUM A1 (post-hoc deviation): RandTrig and RFF = p-1 random features + constant column. Registered result: report_T2T3.txt")
D = L.load_data(); y = D.RV.values
YT, YV = y[L.WI:], y[451:L.WI]
LOSS = {"primary": lambda p, t: (p - t) ** 2 * L.DIF ** 2, "secondary": L.qlike_var_loss}
AUTH = {"QR1": 0, "QR2": 1}
P_OF = {"QR1": 10, "QR2": 20}
rng_seed = 20261112


from stats_lib import compare, holm, verdict, table


def pick(arr_va, arr_te, how):
    """arr_*: (100, n). Returns the row chosen on validation ('val') or on test ('test'), plus its index."""
    if how == "val":
        i = int(np.argmin(((arr_va - YV) ** 2).mean(1)))
    else:
        i = int(np.argmin(((arr_te - YT) ** 2).mean(1)))
    return arr_te[i], i


for design in ("QR1", "QR2"):
    p = P_OF[design]
    QV = np.array([P[f"{design}_{i:03d}_va"] for i in range(100)])
    QT = np.array([P[f"{design}_{i:03d}_te"] for i in range(100)])
    print("=" * 100); print(f"{design}  ({p} readout features; controls of size {p})"); print("=" * 100)
    mse = ((QT - YT) ** 2).mean(1) * L.DIF ** 2
    a = AUTH[design]
    print(f"\nT2a  test MSE over 100 instances: median {np.median(mse):.4f}, 5-95% [{np.percentile(mse, 5):.4f}, {np.percentile(mse, 95):.4f}], "
          f"min {mse.min():.4f}, max {mse.max():.4f}")
    print(f"     authors' instance {a + 1}: MSE {mse[a]:.4f}, rank {int((mse < mse[a]).sum()) + 1} of 100")
    ar3, har = P["AR3_te"], P["HAR_te"]
    for nm, b in (("AR3", ar3), ("HAR", har)):
        print(f"     {nm} test MSE {((b - YT) ** 2).mean() * L.DIF ** 2:.4f}; instances beating it: {int((mse < ((b - YT) ** 2).mean() * L.DIF ** 2).sum())} / 100")
    qv, iv = pick(QV, QT, "val"); qt, it = pick(QV, QT, "test")
    print(f"     validation-selected instance {iv + 1}: test MSE {mse[iv]:.4f}; test-selected instance {it + 1}: {mse[it]:.4f}")

    for lname, lf in LOSS.items():
        print(f"\n--- loss: {lname} ---")
        rows = [(f"QR_val vs {nm}", *compare(lf(qv, YT), lf(b, YT))) for nm, b in (("AR3", ar3), ("HAR", har))]
        table(rows, "T2b honest selection")
        print("\nT2c Hansen SPA over the 100 instances (consistent p-value)")
        M = np.column_stack([lf(QT[i], YT) for i in range(100)])
        for nm, b in (("AR3", ar3), ("HAR", har)):
            spa = SPA(lf(b, YT), M, block_size=6, reps=10000, bootstrap="stationary", seed=rng_seed)
            spa.compute()
            print(f"    vs {nm}: p = {float(spa.pvalues['consistent']):.4f}")
        rows = []
        for kind in ("ESN", "RandTrig", "RFF"):
            cv, ct = P[f"{design}_{kind}_va"], P[f"{design}_{kind}_te"]
            rows.append((f"QR_testbest vs {kind}-{p} testbest", *compare(lf(qt, YT), lf(pick(cv, ct, 'test')[0], YT))))
        table(rows, "T2d selection-matched, both best-of-100 on TEST (mimics the paper)")
        rows = []
        for kind in ("ESN", "RandTrig", "RFF"):
            cv, ct = P[f"{design}_{kind}_va"], P[f"{design}_{kind}_te"]
            rows.append((f"QR_val vs {kind}-{p} val-selected", *compare(lf(qv, YT), lf(pick(cv, ct, 'val')[0], YT))))
        table(rows, "T2d selection-matched, both selected on VALIDATION")
        ctrl = {f"ESNtuned-{p}": pick(P[f"{design}_ESNtuned_va"], P[f"{design}_ESNtuned_te"], "val")[0],
                f"RandTrig-{p}": pick(P[f"{design}_RandTrig_va"], P[f"{design}_RandTrig_te"], "val")[0],
                f"RFF-{p}": pick(P[f"{design}_RFF_va"], P[f"{design}_RFF_te"], "val")[0],
                "Trig2": P[f"{design}_Trig2_te"][0], "QR_J0": P[f"{design}_J0_te"], "AR3": ar3, "HAR": har}
        rows = [(f"QR_val vs {nm}", *compare(lf(qv, YT), lf(b, YT))) for nm, b in ctrl.items()]
        ph = table(rows, "T3 size-matched and surrogate controls (Holm over 7)")
        if lname == "primary":
            survives_t3 = all(r[1] < 1 and h < 0.05 for r, h in zip(rows, ph))
        Lm = pd.DataFrame({"QR_val": lf(qv, YT), **{nm: lf(b, YT) for nm, b in ctrl.items()}})
        mcs = MCS(Lm, size=0.05, reps=10000, block_size=6, method="R", seed=rng_seed); mcs.compute()
        print("\nT3 95% Model Confidence Set: included =", list(mcs.included))
        print("   MCS p-values:", {k: round(float(v), 3) for k, v in mcs.pvalues["Pvalue"].items()})
        for nm in ("QR_val", *ctrl):
            arr = Lm[nm].values
            print(f"   mean loss {nm:14s} {arr.mean():.5f}")
    print(f"\nTrig2 ridge lambda chosen on validation: {float(P[f'{design}_Trig2_lam'])}")
    print(f"\nDECISION ({design}, primary loss): T3 all-better-after-Holm = {survives_t3}; "
          f"T2c must also reject for AR3 and HAR (see above). Claim survives only if both hold.")
