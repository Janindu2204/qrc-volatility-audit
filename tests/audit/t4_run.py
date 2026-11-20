"""Test 4 (prereg_li_audit_T4T6.json): (a) train-window-only input scaling; (b) forward feature selection scored on validation vs on test.
Usage: python t4_run.py [workers] [b]  ("b": selection only, reusing preds_T4a.npz). Output: out/preds_T4a.npz, out/T4b_selection.json."""
import os, sys, json, hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CANDS = ["DP", "EP", "MKT", "SMB", "HML", "TB", "DEF", "IP", "INF", "RV", "STR", "RV_q", "RV_a"]
RANGE = {"DP": (0, 1), "EP": (0, 1), "TB": (0, 1), "DEF": (0, 1), "MKT": (-1, 1), "SMB": (-1, 1), "HML": (-1, 1), "STR": (-1, 1),
         "RV": (-1, 0), "RV_q": (-1, 0), "RV_a": (-1, 0)}
DES = {"QR1": (0, 1, "F_QR1"), "QR2": (1, 2, "F_QR2")}


def rescaled(L):
    D = L.load_data().copy()
    for c, (a, b) in RANGE.items():
        lo, hi = D[c].iloc[:L.WI].min(), D[c].iloc[:L.WI].max()
        D[c] = np.clip(a + (b - a) * (D[c] - lo) / (hi - lo), a, b)
    return D


def readouts(L, F, y):
    te = L.rolling_ridge(F, y)
    va = L.rolling_ridge(F[:L.WI], y[:L.WI], wi=451, L=120)
    return va, te


def t4a_job(args):
    design, inst = args
    import li_lib as L
    D = rescaled(L); _, v, fn = DES[design]
    F = L.reservoir_features(D[getattr(L, fn)].values, L.load_couplings()[inst], vnode=v)
    return design, inst, readouts(L, F, L.load_data().RV.values)


def t4b_eval(args):
    """One forward-selection candidate: validation and test MSE (scaled units) of the readout on input columns cols."""
    design, cols = args
    import li_lib as L
    D = L.load_data(); y = D.RV.values; inst, v, _ = DES[design]
    F = L.reservoir_features(D[list(cols)].values, L.load_couplings()[inst], vnode=v)
    va, te = readouts(L, F, y)
    return design, cols, float(np.mean((va - y[451:L.WI]) ** 2)), float(np.mean((te - y[L.WI:]) ** 2))


def t4b(pool):
    """Forward selection of 7 of 13 inputs per design, scored on validation or on test, all four paths stepped together.
    Order of each feature set follows the order it was added, as in a sequential search; results are cached and saved after every step."""
    import time, li_lib as L
    paths = {(d, c): [] for d in DES for c in ("val", "test")}
    cache = {}
    for step in range(7):
        t0 = time.time()
        todo = sorted({(d, tuple(x["added"] for x in p) + (c,)) for (d, _), p in paths.items()
                       for c in CANDS if c not in [x["added"] for x in p]} - set(cache))
        for k, (d, cols, va, te) in enumerate(pool.imap_unordered(t4b_eval, todo), 1):
            cache[(d, cols)] = (va, te)
            if k % 10 == 0 or k == len(todo):
                print(f"T4b step {step + 1}/7: {k}/{len(todo)} evaluated, {time.time() - t0:.0f}s", flush=True)
        for (d, crit), p in paths.items():
            base = tuple(x["added"] for x in p)
            cands = [(cache[(d, base + (c,))], c) for c in CANDS if c not in base]
            (va, te), c = min(cands, key=lambda z: z[0][0] if crit == "val" else z[0][1])
            p.append({"added": c, "criterion_value": va if crit == "val" else te, "test_mse": te * L.DIF ** 2})
        (OUT / "T4b_selection.json").write_text(json.dumps({f"{d}_{c}": p for (d, c), p in paths.items()}, indent=1))
    return {f"{d}_{c}": p for (d, c), p in paths.items()}


if __name__ == "__main__":
    sys.path.insert(0, str(HERE)); os.chdir(HERE); OUT.mkdir(exist_ok=True)
    h = hashlib.sha256((HERE / "prereg_li_audit_T4T6.json").read_bytes()).hexdigest()
    assert h == (HERE / "prereg_li_audit_T4T6.sha256").read_text().split()[0], "registration changed after freezing"
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    only_b = sys.argv[2:] == ["b"]
    with Pool(w, maxtasksperchild=20) as pool:
        if not only_b:
            save = {}
            for design, inst, (va, te) in pool.imap_unordered(t4a_job, [(d, i) for d in DES for i in range(100)]):
                save[f"{design}_{inst:03d}_va"], save[f"{design}_{inst:03d}_te"] = va, te
            np.savez_compressed(OUT / "preds_T4a.npz", **save)
        res = t4b(pool)
    print(json.dumps(res, indent=1))
