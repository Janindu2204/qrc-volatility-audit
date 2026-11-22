"""Descriptive (not registered) figure: cumulative squared-error gain of the reservoir over HAR across the 245 test months,
from saved Test 2 forecasts only. Usage: python make_fig_cumloss.py <output folder>. Writes sanda6.pdf/.png and prints the numbers."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300})
P = np.load(HERE / "out" / "preds_T2T3.npz")
D = L.load_data(); y = D.RV.values; YT, YV = y[L.WI:], y[451:L.WI]
dates = pd.to_datetime(D.Date.values[L.WI:])
se = lambda p: (p - YT) ** 2 * L.DIF ** 2
har = se(P["HAR_te"])

fig, ax = plt.subplots(figsize=(3.45, 2.2))
for d, a, c in (("QR1", 0, "k"), ("QR2", 1, "0.5")):
    QV = np.array([P[f"{d}_{i:03d}_va"] for i in range(100)]); iv = int(np.argmin(((QV - YV) ** 2).mean(1)))
    for i, ls, lab in ((a, "-", "published"), (iv, "--", "validation-selected")):
        g = har - se(P[f"{d}_{i:03d}_te"]); o = np.argsort(-g); top = o[:10]
        q, h = se(P[f"{d}_{i:03d}_te"]), har
        rest = (q.sum() - q[top].sum()) / (h.sum() - h[top].sum())
        print(f"{d} {lab} instance {i + 1}: ratio {q.mean() / h.mean():.3f}; top-10 months share of gain {g[top].sum() / g.sum():.2f}; "
              f"ratio on other 235 months {rest:.3f}; months better {(g > 0).sum()}/245; top 5 {[str(x)[:7] for x in dates[o[:5]]]}")
        ax.plot(dates, np.cumsum(g), ls=ls, color=c, lw=0.9, label=f"{d}, {lab}")
ax.axhline(0, color="0.6", lw=0.5)
ax.set_ylabel("Cumulative SE gain over HAR"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "sanda6.pdf"); fig.savefig(OUT / "sanda6.png"); plt.close(fig)
