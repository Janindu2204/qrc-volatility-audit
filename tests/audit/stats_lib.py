"""Shared statistics for the Li et al. audit: loss ratio with stationary-bootstrap interval, Newey-West Diebold-Mariano, Holm, verdict."""
import numpy as np
from scipy.stats import norm
from arch.bootstrap import StationaryBootstrap

SEED = 20261112


def nw_var(d, lag=4):
    d = d - d.mean(); n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm_p(la, lb, lag=4):
    d = la - lb
    return float(2 * norm.sf(abs(d.mean() / np.sqrt(nw_var(d, lag) / len(d)))))


def compare(la, lb):
    r = la.mean() / lb.mean()
    bs = StationaryBootstrap(6, la, lb, seed=SEED)
    ci = bs.conf_int(lambda a, b: np.array([a.mean() / b.mean()]), reps=5000, method="percentile").ravel()
    return r, ci[0], ci[1], dm_p(la, lb)


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def verdict(r, lo, hi, ph):
    if ph < 0.05:
        return "better" if r < 1 else "worse"
    if lo >= 0.97 and hi <= 1.03:
        return "equivalent"
    return "inconclusive"


def table(rows, title):
    print(f"\n{title}")
    print(f"{'comparison':40s} {'ratio':>7s} {'95% CI':>17s} {'p':>8s} {'p_holm':>8s}  verdict")
    ph = holm([r[4] for r in rows])
    for (nm, r, lo, hi, p), h in zip(rows, ph):
        print(f"{nm:40s} {r:7.4f} [{lo:6.4f}, {hi:6.4f}] {p:8.4f} {h:8.4f}  {verdict(r, lo, hi, h)}")
    return ph
