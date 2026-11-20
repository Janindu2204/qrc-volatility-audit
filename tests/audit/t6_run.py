"""Test 6 (prereg_li_audit_T4T6.json): minimum detectable MSE gain for the Li et al. design (245 months) and the copper transfer (120 months).
Usage: python t6_run.py > report_T6.txt (needs out/preds_T2T3.npz and out/preds_T7.npz).
Supplementary, not registered: python t6_run.py fine > report_T6_fine.txt (1% steps from 5% to 40%)."""
import sys, hashlib
from pathlib import Path
import numpy as np
from arch.bootstrap import StationaryBootstrap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L, t7_run as T
from stats_lib import dm_p

h = hashlib.sha256((HERE / "prereg_li_audit_T4T6.json").read_bytes()).hexdigest()
assert h == (HERE / "prereg_li_audit_T4T6.sha256").read_text().split()[0]
FINE = sys.argv[1:] == ["fine"]
DELTAS = [round(0.01 * k, 2) for k in range(5, 41)] if FINE else [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
if FINE:
    print("SUPPLEMENTARY (not pre-registered): finer delta grid; the registered grid is in report_T6.txt")
REPS = 2000


def errors_li(design):
    P = np.load(HERE / "out" / "preds_T2T3.npz"); y = L.load_data().RV.values; YT, YV = y[L.WI:], y[451:L.WI]
    QV = np.array([P[f"{design}_{i:03d}_va"] for i in range(100)]); QT = np.array([P[f"{design}_{i:03d}_te"] for i in range(100)])
    q = QT[int(np.argmin(((QV - YV) ** 2).mean(1)))]
    return P["HAR_te"] - YT, q - YT


def errors_cu(design):
    P = np.load(HERE / "out" / "preds_T7.npz"); _, S, ntr, _ = T.build("HG"); y = S.RV.values; YT, YV = y[ntr:], y[ntr - 36:ntr]
    QV = np.array([P[f"HG_{design}_{i:03d}_va"] for i in range(100)]); QT = np.array([P[f"HG_{design}_{i:03d}_te"] for i in range(100)])
    q = QT[int(np.argmin(((QV - YV) ** 2).mean(1)))]
    return P[f"HG_{design}_HAR_te"] - YT, q - YT


def power(eb, rho, delta, alpha, seed):
    sd = eb.std(); k = rho * np.sqrt(1 - delta); s = sd * np.sqrt((1 - delta) * (1 - rho ** 2))
    bs = StationaryBootstrap(6, eb, seed=seed); hits = 0
    for i, (data, _) in enumerate(bs.bootstrap(REPS)):
        u = data[0]; u = (u - u.mean()) / u.std()
        ec = k * eb + s * u
        lb, lc = eb ** 2, ec ** 2
        if lc.mean() < lb.mean() and dm_p(lc, lb) < alpha:
            hits += 1
    return hits / REPS


for label, fn in (("Li et al. design, 245 months", errors_li), ("copper transfer, 120 months", errors_cu)):
    for design in ("QR1", "QR2"):
        eb, eq = fn(design); rho = float(np.corrcoef(eb, eq)[0, 1])
        print(f"\n{label}, {design}: n = {len(eb)}, corr(HAR errors, QR_val errors) = {rho:.3f}, observed MSE ratio QR_val/HAR = {np.mean(eq ** 2) / np.mean(eb ** 2):.4f}")
        for alpha, nm in ((0.05, "single test"), (0.05 / 7, "Holm, most significant of 7")):
            pw = [power(eb, rho, d, alpha, 100 + j) for j, d in enumerate(DELTAS)]
            mde = next((d for d, p in zip(DELTAS, pw) if p >= 0.8), None)
            print(f"  {nm:28s} power: " + "  ".join(f"{int(100 * d)}%:{p:.2f}" if d >= 0.01 and (100 * d) % 1 == 0 else f"{100 * d:.1f}%:{p:.2f}" for d, p in zip(DELTAS, pw))
                  + f"   -> 80% MDE = {('%.1f%%' % (100 * mde)) if mde else '> %d%%' % round(100 * DELTAS[-1])}")
