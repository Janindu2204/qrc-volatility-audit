"""Tests 2-3 of prereg_li_audit_T2T3.json: run all 100 QRC instances and the classical controls; save validation and test predictions.
Usage: python t2t3_run.py [workers]. Output: out/preds_T2T3.npz (test rows 571..815, validation rows 451..570)."""
import os, sys, hashlib, json, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VAL_WI, VAL_L = 451, 120
DESIGNS = {"QR1": (1, "F_QR1", 10), "QR2": (2, "F_QR2", 20)}


def check_prereg():
    h = hashlib.sha256((HERE / "prereg_li_audit_T2T3.json").read_bytes()).hexdigest()
    assert h == (HERE / "prereg_li_audit_T2T3.sha256").read_text().split()[0], "registration changed after freezing"


def readouts(L, F, y, lam=1e-8, intercept=False):
    te = L.rolling_ridge(F, y, lam=lam, intercept=intercept)
    va = L.rolling_ridge(F[:L.WI], y[:L.WI], lam=lam, wi=VAL_WI, L=VAL_L, intercept=intercept)
    return va, te


def qr_job(args):
    design, inst = args
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import li_lib as L
    D, Js = L.load_data(), L.load_couplings()
    v, fname, _ = DESIGNS[design]
    J = Js[inst] if inst >= 0 else np.zeros((10, 10))
    F = L.reservoir_features(D[getattr(L, fname)].values, J, vnode=v)
    return design, inst, readouts(L, F, D.RV.values)


def esn_feats(Xw, p, rng, leak, rho, scale):
    """Xw: (n, 3, 7) lag windows, oldest first. Zero initial state, bias input."""
    n, K, d = Xw.shape
    Win = rng.uniform(-1, 1, (p, d + 1)) * scale
    W = rng.normal(size=(p, p)); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h = np.zeros((n, p))
    for k in range(K):
        u = np.hstack([np.ones((n, 1)), Xw[:, k]])
        h = (1 - leak) * h + leak * np.tanh(u @ Win.T + h @ W.T)
    return h


def randtrig_feats(Z, p, rng):
    cols = []
    for _ in range(p):
        m = rng.integers(1, 4); vs = rng.choice(Z.shape[1], m, replace=False)
        f = np.ones(len(Z))
        for v in vs:
            f = f * (np.cos(np.pi * Z[:, v]) if rng.random() < 0.5 else np.sin(np.pi * Z[:, v]))
        cols.append(f)
    return np.column_stack(cols)


def rff_feats(Z, p, rng, g):
    w = rng.normal(0, g, (Z.shape[1], p)); b = rng.uniform(0, 2 * np.pi, p)
    return np.cos(Z @ w + b)


def trig2_feats(Z):
    c, s = np.cos(np.pi * Z), np.sin(np.pi * Z)
    cols = [np.ones(len(Z))] + [c[:, v] for v in range(Z.shape[1])] + [s[:, v] for v in range(Z.shape[1])]
    for u, v in itertools.combinations(range(Z.shape[1]), 2):
        for a in (c, s):
            for b in (c, s):
                cols.append(a[:, u] * b[:, v])
    return np.column_stack(cols)


def control_job(args):
    design, = args
    import li_lib as L
    D = L.load_data(); y = D.RV.values
    _, fname, p = DESIGNS[design]
    feats = getattr(L, fname)
    Z = L.lagged_window(D, feats)                         # (n, 21) = [x_{t-3}, x_{t-2}, x_{t-1}]
    Xw = Z.reshape(len(Z), 3, len(feats))
    res = {}
    seed0 = {"QR1": 1000, "QR2": 2000}[design]
    for kind in ("ESN", "RandTrig", "RFF"):
        va_l, te_l = [], []
        for i in range(100):
            rng = np.random.default_rng(seed0 + i)
            if kind == "ESN":
                F = esn_feats(Xw, p, rng, 0.6, 0.9, 0.1)
            elif kind == "RandTrig":
                F = randtrig_feats(Z, p, rng)
            else:
                best = None
                for g in (0.3, 1.0, 3.0):
                    Fg = rff_feats(Z, p, np.random.default_rng(seed0 + i), g)
                    va, te = readouts(L, Fg, y)
                    m = np.mean((va - y[VAL_WI:L.WI]) ** 2)
                    if best is None or m < best[0]:
                        best = (m, va, te)
                va_l.append(best[1]); te_l.append(best[2]); continue
            va, te = readouts(L, F, y); va_l.append(va); te_l.append(te)
        res[kind] = (np.array(va_l), np.array(te_l))
    va_l, te_l = [], []                                   # tuned ESN: 27 configs per instance, chosen on validation
    for i in range(100):
        best = None
        for leak, rho, sc in itertools.product((0.3, 0.6, 1.0), (0.5, 0.9, 1.2), (0.1, 0.5, 1.0)):
            F = esn_feats(Xw, p, np.random.default_rng(seed0 + i), leak, rho, sc)
            va, te = readouts(L, F, y)
            m = np.mean((va - y[VAL_WI:L.WI]) ** 2)
            if best is None or m < best[0]:
                best = (m, va, te)
        va_l.append(best[1]); te_l.append(best[2])
    res["ESNtuned"] = (np.array(va_l), np.array(te_l))
    T2 = trig2_feats(Z); best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        va, te = readouts(L, T2, y, lam=lam)
        m = np.mean((va - y[VAL_WI:L.WI]) ** 2)
        if best is None or m < best[0]:
            best = (m, va, te, lam)
    res["Trig2"] = (best[1][None], best[2][None]); res["Trig2_lam"] = best[3]
    return design, res


def linear_baselines():
    import li_lib as L, pandas as pd
    D = L.load_data(); y = D.RV.values; n = len(y); s = pd.Series(y)
    l1, l2, l3 = (s.shift(k).fillna(0).values for k in (1, 2, 3))
    q = s.rolling(3).mean().shift(1).fillna(0).values; a = s.rolling(12).mean().shift(1).fillna(0).values
    out = {}
    for nm, X in (("AR3", np.c_[l1, l2, l3]), ("HAR", np.c_[l1, q, a])):
        out[nm] = readouts(L, np.c_[X, np.ones(n)], y, lam=0.0)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(HERE)); os.chdir(HERE); check_prereg(); OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    jobs = [(d, i) for d in DESIGNS for i in list(range(100)) + [-1]]
    save = {}
    with Pool(workers) as pool:
        ctrl = pool.map_async(control_job, [(d,) for d in DESIGNS])
        for k, (design, inst, (va, te)) in enumerate(pool.imap_unordered(qr_job, jobs)):
            tag = f"{design}_J0" if inst < 0 else f"{design}_{inst:03d}"
            save[tag + "_va"], save[tag + "_te"] = va, te
            if k % 20 == 0:
                print("qr done", k + 1, "/", len(jobs), flush=True)
        for design, res in ctrl.get():
            for kind, v in res.items():
                if kind == "Trig2_lam":
                    save[f"{design}_Trig2_lam"] = np.array(v)
                else:
                    save[f"{design}_{kind}_va"], save[f"{design}_{kind}_te"] = v
    for nm, (va, te) in linear_baselines().items():
        save[nm + "_va"], save[nm + "_te"] = va, te
    np.savez_compressed(OUT / "preds_T2T3.npz", **save)
    print("saved", len(save), "arrays")
