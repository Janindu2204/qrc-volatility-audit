"""Test 5 (prereg_li_audit_T4T6.json): (a) per-input sweep check of the multilinear-trig proposition; (b) holdout R^2 of the reservoir
readout on order<=1 and order<=2 multilinear trig bases. Usage: python t5_run.py > report_T5.txt"""
import os, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); os.chdir(HERE)
import li_lib as L, t2t3_run as R

h = hashlib.sha256((HERE / "prereg_li_audit_T4T6.json").read_bytes()).hexdigest()
assert h == (HERE / "prereg_li_audit_T4T6.sha256").read_text().split()[0]
D, Js = L.load_data(), L.load_couplings()
rng = np.random.default_rng(55)
DES = {"QR1": (0, 1, L.F_QR1), "QR2": (1, 2, L.F_QR2)}


def win_feats(W, J, v, UdU):
    """W: (m, 3, 7) windows -> (m, 10v) readouts, chunked."""
    out = []
    for s in range(0, len(W), 800):
        w = W[s:s + 800]; m = len(w)
        X = w.reshape(3 * m, w.shape[2]); rows = np.arange(3, 3 * m + 1, 3)
        X = np.vstack([X, np.zeros((1, X.shape[1]))])          # row 3j+3 reads rows 3j..3j+2
        out.append(L.reservoir_features(X, J, vnode=v, rows=rows, UdU=UdU)[rows])
    return np.vstack(out)


print("T5a sweep check: max residual RMS of a + b cos(pi x) + c sin(pi x) per input")
for design, (inst, v, feats) in DES.items():
    J = Js[inst]; H = L.hamiltonian(J); UdU = (expm(-1j * H), expm(-1j / v * H))
    lo, hi = D[feats].min().values, D[feats].max().values
    worst = 0.0
    for lag in range(3):
        for f in range(7):
            base = rng.uniform(lo, hi, (3, 7))
            xs = np.linspace(lo[f], hi[f], 64)
            W = np.repeat(base[None], 64, 0); W[:, lag, f] = xs
            Y = win_feats(W, J, v, UdU)
            B = np.c_[np.ones(64), np.cos(np.pi * xs), np.sin(np.pi * xs)]
            res = Y - B @ np.linalg.lstsq(B, Y, rcond=None)[0]
            worst = max(worst, float(np.sqrt((res ** 2).mean(0)).max()))
    print(f"  {design} instance {inst + 1}: worst residual RMS over 21 inputs x {10 * v} features = {worst:.2e}")

print("\nT5b holdout R^2 (median over readout features; min in brackets)")
for design, (inst, v, feats) in DES.items():
    insts = [inst] + list(rng.choice([i for i in range(100) if i != inst], 9, replace=False))
    lo, hi = D[feats].min().values, D[feats].max().values
    Xall = D[feats].values
    for dist in ("uniform", "empirical"):
        if dist == "uniform":
            W = rng.uniform(lo, hi, (16000, 3, 7))
        else:
            st = rng.integers(0, len(Xall) - 3, 16000); W = np.stack([Xall[st + k] for k in range(3)], 1)
        Z = W.reshape(16000, 21)
        T = R.trig2_feats(Z); T1 = T[:, :43]
        r1s, r2s = [], []
        for i in insts:
            J = Js[i]; H = L.hamiltonian(J); UdU = (expm(-1j * H), expm(-1j / v * H))
            Y = win_feats(W, J, v, UdU)
            for Bm, acc in ((T1, r1s), (T, r2s)):
                c = np.linalg.lstsq(Bm[:12000], Y[:12000], rcond=None)[0]
                e = Y[12000:] - Bm[12000:] @ c
                r2 = 1 - (e ** 2).sum(0) / ((Y[12000:] - Y[12000:].mean(0)) ** 2).sum(0)
                acc.append(r2)
        r1s, r2s = np.concatenate(r1s), np.concatenate(r2s)
        print(f"  {design} {dist:9s}: order<=1 R^2 median {np.median(r1s):.4f} [min {r1s.min():.4f}] | "
              f"order<=2 R^2 median {np.median(r2s):.4f} [min {r2s.min():.4f}]  ({len(insts)} instances)", flush=True)
