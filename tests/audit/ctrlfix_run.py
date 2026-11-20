"""Addendum A1 (addendum_A1_controls_2026-11-13.json): RandTrig and RFF controls rebuilt as p-1 random features plus one constant column.
The readout has no intercept; QR features carry a non-zero mean that acts as one, the zero-mean controls did not.
Same seeds, same validation selection and RFF grid as the registered runs. Usage: python ctrlfix_run.py [workers]. Output: out/preds_ctrlfix.npz."""
import os, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
const = lambda F: np.c_[F, np.ones(len(F))]


def li_job(design):
    import li_lib as L, t2t3_run as R
    D = L.load_data(); y = D.RV.values
    _, fname, p = R.DESIGNS[design]
    Z = L.lagged_window(D, getattr(L, fname)); yv = y[R.VAL_WI:L.WI]
    seed0 = {"QR1": 1000, "QR2": 2000}[design]
    return [(design, "Li", *runs(Z, p, seed0, lambda F: R.readouts(L, F, y), yv, R))]


def cu_job(args):
    asset, design = args
    import li_lib as L, t2t3_run as R, t7_run as T
    _, S, ntr, _ = T.build(asset); y = S.RV.values
    nval = int((S.index >= pd.Period(T.VAL0, "M")).sum() - (S.index >= pd.Period(T.TEST0, "M")).sum())
    _, p = T.DESIGNS[design]
    Z = L.lagged_window(S, list(S.columns))
    seed0 = 10000 * (T.ASSETS.index(asset) + 1) + (0 if design == "QR1" else 5000)
    return [(design, asset, *runs(Z, p, seed0, lambda F: T.readouts(F, y, ntr, nval), y[ntr - nval:ntr], R))]


def runs(Z, p, seed0, ro, yv, R):
    out = {}
    for kind in ("RandTrig", "RFF"):
        va_l, te_l = [], []
        for i in range(100):
            if kind == "RandTrig":
                va, te = ro(const(R.randtrig_feats(Z, p - 1, np.random.default_rng(seed0 + i))))
            else:
                best = None
                for g in (0.3, 1.0, 3.0):
                    va, te = ro(const(R.rff_feats(Z, p - 1, np.random.default_rng(seed0 + i), g)))
                    m = np.mean((va - yv) ** 2)
                    if best is None or m < best[0]:
                        best = (m, va, te)
                va, te = best[1], best[2]
            va_l.append(va); te_l.append(te)
        out[kind] = (np.array(va_l), np.array(te_l))
    return (out,)


def job(a):
    return li_job(a[1]) if a[0] == "Li" else cu_job(a)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE)); os.chdir(HERE)
    import t7_run as T
    jobs = [("Li", d) for d in ("QR1", "QR2")] + [(a, d) for a in T.ASSETS for d in ("QR1", "QR2")]
    save = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for k, res in enumerate(pool.imap_unordered(job, jobs), 1):
            for design, where, out in res:
                pre = f"{design}" if where == "Li" else f"{where}_{design}"
                for kind, (va, te) in out.items():
                    save[f"{pre}_{kind}_va"], save[f"{pre}_{kind}_te"] = va, te
            print(f"ctrlfix {k}/{len(jobs)}", flush=True)
    np.savez_compressed(OUT / "preds_ctrlfix.npz", **save)
    print("saved", len(save))
