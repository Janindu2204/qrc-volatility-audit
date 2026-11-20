"""Own re-implementation of the quantum reservoir (QR1/QR2) of Li et al., Phys. Rev. Research 8, 023028 (2026), plus the audit controls.

Model (read from the authors' Time_series.jl): 10 qubits, H = sum_{i<j} J_ij X_i X_j + sum_i Z_i, U = exp(-i tau H), tau = 1, K = 3 lags.
For each forecast time l the 3 hidden qubits start in |000>; for k = 3, 2, 1 the 7 input qubits are freshly prepared as RY(pi x_{l-k})|0>,
joined with the hidden state, evolved; for k > 1 the input qubits are traced out. At k = 1 the joint state is evolved V times by
exp(-i tau/V H) and <Z_q> of all 10 qubits is read after each sub-step (V = 1: QR1, 10 features; V = 2: QR2, 20 features).
Readout: rolling ridge, window 571 months, lambda 1e-8, no intercept, 245 one-step forecasts.
Qubit q is bit q of the basis index (input qubits 0..6, hidden 7..9), the convention that reproduced the authors' output in an independent
Qiskit replication. The hidden state is kept as rank <= 8 factors A (rho_h = A A^H), so each step evolves at most 8 vectors of size 1024.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import expm

DATA = Path(__file__).resolve().parents[2] / "data" / "li2026"
MAX_RV, MIN_RV = -1.2543188032019446, -4.7722718186046515      # from Time_series.jl, used to undo the authors' scaling
DIF = MAX_RV - MIN_RV
NQ, NIN = 10, 7
TOTAL, L_OOS = 816, 245
WI = TOTAL - L_OOS                                                # 571-month rolling window
F_QR1 = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]
F_QR2 = ["RV", "MKT", "STR", "RV_q", "EP", "INF", "DEF"]


def load_data():
    return pd.read_csv(DATA / "Data.CSV")


def load_couplings():
    import h5py
    with h5py.File(DATA / "coeff_10.jld2", "r") as f:
        m = f["ms"][:]                                            # stored (10, 10, 100) = Julia (100, 10, 10) column-major
    return np.ascontiguousarray(np.transpose(m, (2, 1, 0)))


def load_ref():
    return pd.read_csv(DATA / "predict_result.csv")


def hamiltonian(J, field=1.0):
    dim = 2 ** NQ
    b = np.arange(dim)
    bits = (b[:, None] >> np.arange(NQ)) & 1
    H = np.zeros((dim, dim))
    H[b, b] = field * (1 - 2 * bits).sum(1)
    for i in range(NQ):
        for j in range(i + 1, NQ):
            if J[i, j] != 0:
                H[b, b ^ ((1 << i) | (1 << j))] += J[i, j]
    return H


def _input_states(X):
    """X: (n, 7) raw scaled features -> (n, 128) product states RY(pi x)|0>, qubit q = bit q."""
    c, s = np.cos(np.pi * X / 2), np.sin(np.pi * X / 2)
    st = np.ones((X.shape[0], 1))
    for q in range(X.shape[1] - 1, -1, -1):                       # highest qubit first so qubit 0 ends as the least significant bit
        st = np.einsum("na,nb->nab", st, np.stack([c[:, q], s[:, q]], 1)).reshape(X.shape[0], -1)
    return st


def reservoir_features(X, J, vnode=1, K=3, tau=1.0, field=1.0, rows=None, UdU=None):
    """X: (n, nin). Returns (n, 10*vnode); rows < K are zero (as in the authors' code). rows: only compute these forecast rows."""
    n, nin = X.shape
    dh, di = 2 ** (NQ - nin), 2 ** nin
    if UdU is None:
        H = hamiltonian(J, field)
        UdU = expm(-1j * tau * H), expm(-1j * tau / vnode * H)
    U, dU = UdU
    psi_in = _input_states(X).astype(complex)
    rows = np.arange(K, n) if rows is None else np.asarray(rows)
    B = len(rows)
    A = np.zeros((B, dh, 1), complex); A[:, 0, 0] = 1.0         # hidden |0...0>
    zsign = 1 - 2 * ((np.arange(2 ** NQ)[:, None] >> np.arange(NQ)) & 1)
    out = np.zeros((n, NQ * vnode))
    for k in range(K, 0, -1):
        p = psi_in[rows - k]                                      # (B, 2^nin)
        r = A.shape[2]
        V = np.einsum("bhr,bi->brhi", A, p).reshape(B * r, 2 ** NQ)   # joint vectors, index h*2^nin + i
        if k != 1:
            V = (V @ U.T).reshape(B, r, dh, di)
            A = np.transpose(V, (0, 2, 1, 3)).reshape(B, dh, r * di)  # rho_h = A A^H after tracing out the inputs
            if A.shape[2] > dh:
                w, Q = np.linalg.eigh(np.einsum("bhk,bgk->bhg", A, A.conj()))
                A = Q * np.sqrt(np.clip(w, 0, None))[:, None, :]
        else:
            for v in range(vnode):
                V = V @ dU.T
                prob = (np.abs(V) ** 2).reshape(B, r, -1).sum(1)
                out[rows, v * NQ:(v + 1) * NQ] = prob @ zsign
    return out


def rolling_ridge(F, y, lam=1e-8, wi=WI, L=L_OOS, intercept=False):
    """Authors' readout: for j = 1..L train on rows j-1 .. j+wi-2 (0-based), predict row j+wi-1."""
    if intercept:
        F = np.hstack([F, np.ones((F.shape[0], 1))])
    p = F.shape[1]
    pred = np.zeros(L)
    for j in range(L):
        Xtr, ytr = F[j:j + wi], y[j:j + wi]
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(p), Xtr.T @ ytr)
        pred[j] = F[j + wi] @ w
    return pred


def to_logrv(z):
    return (z + 1) * DIF + MIN_RV


def mse_paper(pred, y):
    return float(np.mean((pred - y) ** 2) * DIF ** 2)


def qlike_paper(pred, y):
    """compute_qlike in Time_series.jl: summed over the sample, applied to |log RV| (not to variance)."""
    f, a = np.abs(to_logrv(pred)), np.abs(to_logrv(y))
    r = a / f
    return float(np.sum(r - np.log(r) - 1))


def qlike_var_loss(pred, y):
    """Standard per-period QLIKE on variance: the scaled RV column is log(sqrt(sum r^2)), so variance = exp(2 * logRV)."""
    hf, ha = np.exp(2 * to_logrv(pred)), np.exp(2 * to_logrv(y))
    return ha / hf - np.log(ha / hf) - 1


def lagged_window(D, cols, K=3):
    """Flattened inputs [x_{t-3}, x_{t-2}, x_{t-1}] for each row t (zeros for t < K), shape (n, K*len(cols))."""
    X = D[cols].values.astype(float); n = len(X)
    Z = np.zeros((n, K * X.shape[1]))
    for t in range(K, n):
        Z[t] = X[t - K:t][::1].reshape(-1)
    return Z
