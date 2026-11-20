"""Test 7 (prereg_li_audit_T7.json): Li et al.'s QRC design transferred to monthly commodity RV. Usage: python t7_run.py [workers]."""
import os, sys, hashlib, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
ASSETS = ["HG", "CL", "GC", "HO", "NG", "ZC", "ZS", "ZW"]
TEST0, VAL0, END = "2015-01", "2012-01", "2024-12"
DESIGNS = {"QR1": (1, 10), "QR2": (2, 20)}
KINDS = {"RV": "rv", "RV_q": "rv", "RV_a": "rv", "RET": "ret", "DRV": "pos", "VIX": "pos", "DXY": "ret"}


def daily_returns(a):
    if a == "HG":
        d = pd.read_csv(ROOT / "data/daily_long/flagged_v2/HG_F_1d.csv", parse_dates=["date"]).set_index("date")
        return d["ret_clean"].dropna()
    if a in ("VIX", "DXY"):
        f = {"VIX": "_VIX_1d.csv", "DXY": "DX-Y.NYB_1d.csv"}[a]
        d = pd.read_csv(ROOT / "data/daily_long" / f, parse_dates=["date"]).set_index("date")
        return d["close"]
    d = pd.read_csv(ROOT / f"data/daily_multi/{a}.csv", parse_dates=["date"]).set_index("date")
    return np.log(d["close"]).diff().dropna()


def monthly_rv(r):
    s = (r ** 2).groupby(r.index.to_period("M")).sum()
    return np.log(np.sqrt(s.where(s > 0)))


def build(asset):
    r = daily_returns(asset)
    rv = monthly_rv(r)
    ret = r.groupby(r.index.to_period("M")).sum()
    drv = monthly_rv(daily_returns("HG" if asset == "CL" else "CL"))
    vix = np.log(daily_returns("VIX").groupby(daily_returns("VIX").index.to_period("M")).mean())
    dx = daily_returns("DXY"); dxy = np.log(dx.groupby(dx.index.to_period("M")).last()).diff()
    M = pd.DataFrame({"RV": rv, "RET": ret, "DRV": drv, "VIX": vix, "DXY": dxy})
    M["RV_q"] = M.RV.rolling(3).mean(); M["RV_a"] = M.RV.rolling(12).mean()
    M = M[(M.index >= pd.Period("2000-10", "M")) & (M.index <= pd.Period(END, "M"))].dropna()
    assert (M.index[1:] - M.index[:-1]).map(lambda x: x.n).max() == 1, f"{asset}: missing months"
    cols = ["RV", "MKT_like", "DRV", "VIX", "RV_q", "RV_a", "DXY"]
    M = M.rename(columns={"RET": "MKT_like"})[cols]
    ntr = int((M.index < pd.Period(TEST0, "M")).sum())
    S = M.copy(); scale = {}
    for c in cols:
        kind = KINDS.get(c, "ret") if c != "MKT_like" else "ret"
        lo, hi = M[c].iloc[:ntr].min(), M[c].iloc[:ntr].max()
        a, b = {"rv": (-1, 0), "ret": (-1, 1), "pos": (0, 1)}[kind]
        S[c] = np.clip(a + (b - a) * (M[c] - lo) / (hi - lo), a, b); scale[c] = (lo, hi, a, b)
    return M, S, ntr, scale


def readouts(F, y, ntr, nval, lam=1e-8, intercept=False):
    import li_lib as L
    n = len(y); Lte = n - ntr
    te = L.rolling_ridge(F, y, lam=lam, wi=ntr, L=Lte, intercept=intercept)
    va = L.rolling_ridge(F[:ntr], y[:ntr], lam=lam, wi=ntr - nval, L=nval, intercept=intercept)
    return va, te


def qr_job(args):
    asset, design, inst = args
    import li_lib as L
    _, S, ntr, _ = build(asset); y = S.RV.values
    nval = int((S.index >= pd.Period(VAL0, "M")).sum() - (S.index >= pd.Period(TEST0, "M")).sum())
    v, _ = DESIGNS[design]
    J = L.load_couplings()[inst] if inst >= 0 else np.zeros((10, 10))
    F = L.reservoir_features(S.values, J, vnode=v)
    return asset, design, inst, readouts(F, y, ntr, nval)


def control_job(args):
    asset, design = args
    import li_lib as L, t2t3_run as R
    M, S, ntr, _ = build(asset); y = S.RV.values
    nval = int((S.index >= pd.Period(VAL0, "M")).sum() - (S.index >= pd.Period(TEST0, "M")).sum())
    _, p = DESIGNS[design]
    Z = L.lagged_window(S, list(S.columns)); Xw = Z.reshape(len(Z), 3, S.shape[1])
    yv = y[ntr - nval:ntr]
    vm = lambda va: np.mean((va - yv) ** 2)
    res = {}; seed0 = 10000 * (ASSETS.index(asset) + 1) + (0 if design == "QR1" else 5000)
    for kind in ("RandTrig", "RFF", "ESNtuned"):
        va_l, te_l = [], []
        for i in range(100):
            if kind == "RandTrig":
                va, te = readouts(R.randtrig_feats(Z, p, np.random.default_rng(seed0 + i)), y, ntr, nval)
            else:
                grid = [(g,) for g in (0.3, 1.0, 3.0)] if kind == "RFF" else list(itertools.product((0.3, 0.6, 1.0), (0.5, 0.9, 1.2), (0.1, 0.5, 1.0)))
                best = None
                for g in grid:
                    rng = np.random.default_rng(seed0 + i)
                    F = R.rff_feats(Z, p, rng, *g) if kind == "RFF" else R.esn_feats(Xw, p, rng, *g)
                    va, te = readouts(F, y, ntr, nval)
                    if best is None or vm(va) < best[0]:
                        best = (vm(va), va, te)
                va, te = best[1], best[2]
            va_l.append(va); te_l.append(te)
        res[kind] = (np.array(va_l), np.array(te_l))
    T2 = R.trig2_feats(Z); best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        va, te = readouts(T2, y, ntr, nval, lam=lam)
        if best is None or vm(va) < best[0]:
            best = (vm(va), va, te)
    res["Trig2"] = (best[1][None], best[2][None])
    s = pd.Series(M.RV.values)
    l1, l2, l3 = (s.shift(k).bfill().values for k in (1, 2, 3))
    q = s.rolling(3).mean().shift(1).bfill().values; a = s.rolling(12).mean().shift(1).bfill().values
    lo, hi, A, B = build(asset)[3]["RV"]
    sc = lambda z: A + (B - A) * (z - lo) / (hi - lo)
    for nm, X in (("AR3", np.c_[sc(l1), sc(l2), sc(l3)]), ("HAR", np.c_[sc(l1), sc(q), sc(a)])):
        res[nm] = readouts(np.c_[X, np.ones(len(y))], y, ntr, nval, lam=0.0)
    return asset, design, res


if __name__ == "__main__":
    sys.path.insert(0, str(HERE)); os.chdir(HERE); OUT.mkdir(exist_ok=True)
    h = hashlib.sha256((HERE / "prereg_li_audit_T7.json").read_bytes()).hexdigest()
    assert h == (HERE / "prereg_li_audit_T7.sha256").read_text().split()[0], "registration changed after freezing"
    for a in ASSETS:
        M, S, ntr, _ = build(a); print(a, len(M), "months", M.index[0], M.index[-1], "train", ntr, flush=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(a, d, i) for a in ASSETS for d in DESIGNS for i in list(range(100)) + [-1]]
    save = {}
    with Pool(workers) as pool:
        ctrl = pool.map_async(control_job, [(a, d) for a in ASSETS for d in DESIGNS])
        for k, (a, d, inst, (va, te)) in enumerate(pool.imap_unordered(qr_job, jobs, chunksize=4)):
            tag = f"{a}_{d}_J0" if inst < 0 else f"{a}_{d}_{inst:03d}"
            save[tag + "_va"], save[tag + "_te"] = va, te
            if k % 100 == 0:
                print("qr done", k + 1, "/", len(jobs), flush=True)
        for a, d, res in ctrl.get():
            for kind, (va, te) in res.items():
                save[f"{a}_{d}_{kind}_va"], save[f"{a}_{d}_{kind}_te"] = va, te
    np.savez_compressed(OUT / "preds_T7.npz", **save)
    print("saved", len(save))
