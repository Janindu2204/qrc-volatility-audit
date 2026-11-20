"""Score Test 8 as registered in prereg_li_audit_T8.json. Run in .venv: python t8_score.py > report_T8.txt
Needs out/T8_pilot_hw.npz, out/T8_t1sim.npz and, for the forecast part, out/T8_full_hw.npz."""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L
from stats_lib import compare, table

D = np.load(L.DATA / "t8_inputs.npz"); EX, y = D["F_exact"], D["y"]
T1 = np.zeros_like(EX); t = np.load(HERE / "out" / "T8_t1sim.npz"); T1[t["rows"]] = t["F"]
print("Jobs:", json.dumps(json.loads((HERE / "out" / "T8_jobs.json").read_text()), default=str)[:600], "\n")

for mode in ("pilot", "full"):
    f = HERE / "out" / f"T8_{mode}_hw.npz"
    if not f.exists():
        print(f"{mode}: no hardware file"); continue
    h = np.load(f); rows, HW, shots = h["rows"], h["F_hw"], int(h["shots"])
    print(f"== {mode}: {len(rows)} rows, {shots} shots (shot-noise scale 1/sqrt(shots) = {1 / np.sqrt(shots):.3f})")
    for nm, ref in (("T1SIM", T1[rows]), ("EXACT", EX[rows])):
        e = np.abs(HW - ref)
        cor = [np.corrcoef(HW[:, q], ref[:, q])[0, 1] for q in range(10)]
        print(f"   HW vs {nm}: mean |dZ| {e.mean():.4f}, max {e.max():.4f}; per-feature correlation median {np.median(cor):.3f} [min {min(cor):.3f}]")
    e = np.abs(T1[rows] - EX[rows]); print(f"   T1SIM vs EXACT (Trotter error only): mean |dZ| {e.mean():.4f}, max {e.max():.4f}")
    print(f"   mean |<Z>| on hardware {np.abs(HW).mean():.4f} vs T1SIM {np.abs(T1[rows]).mean():.4f} (shrinkage toward 0 = depolarising noise)")
    if mode == "full":
        F = np.zeros_like(EX); F[rows] = HW
        YT = y[L.WI:]; lf = lambda p: (p - YT) ** 2 * L.DIF ** 2
        pr = {nm: L.rolling_ridge(G, y) for nm, G in (("HW", F), ("T1SIM", T1), ("EXACT", EX))}
        har = np.load(HERE / "out" / "preds_T2T3.npz")["HAR_te"]
        for nm, p in pr.items():
            print(f"   test MSE {nm}: {lf(p).mean():.4f}")
        print(f"   test MSE HAR: {lf(har).mean():.4f}")
        rows_ = [(f"HW vs {nm}", *compare(lf(pr['HW']), lf(b))) for nm, b in (("T1SIM", pr["T1SIM"]), ("EXACT", pr["EXACT"]), ("HAR", har))]
        table(rows_, "Test 8 forecast, primary loss (Holm over 3)")
