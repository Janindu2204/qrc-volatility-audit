"""Figures for the IEEE Access draft, from saved audit outputs only (no new model runs).
Usage: python make_figures.py <output folder>. Writes sanda1.pdf .. sanda5.pdf."""
import sys, re, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_lib as L, t7_run as T
from stats_lib import compare

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300})
W = 3.45
P = np.load(HERE / "out" / "preds_T2T3.npz")
y = L.load_data().RV.values; YT, YV = y[L.WI:], y[451:L.WI]
mse = lambda p: float(((p - YT) ** 2).mean() * L.DIF ** 2)

# Fig. 1: test MSE of the 100 random reservoirs, Li et al. data
fig, axs = plt.subplots(1, 2, figsize=(W * 2, 2.1), sharey=True)
for ax, (d, a) in zip(axs, (("QR1", 0), ("QR2", 1))):
    QT = np.array([P[f"{d}_{i:03d}_te"] for i in range(100)]); QV = np.array([P[f"{d}_{i:03d}_va"] for i in range(100)])
    m = ((QT - YT) ** 2).mean(1) * L.DIF ** 2; iv = int(np.argmin(((QV - YV) ** 2).mean(1)))
    ax.hist(m, bins=20, color="0.75", edgecolor="0.3", lw=0.4)
    for x, ls, lab in ((m[a], "-", f"published instance (rank {int((m < m[a]).sum()) + 1})"), (m[iv], "--", "validation-selected instance"),
                       (mse(P["HAR_te"]), ":", "HAR(1,3,12)"), (mse(P["AR3_te"]), "-.", "AR(3)")):
        ax.axvline(x, color="k", ls=ls, lw=1, label=lab if ls != "-" else "published instance")
    ax.set_title(d); ax.set_xlabel("Test MSE (log RV), 245 months")
axs[0].set_ylabel("Number of instances")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(OUT / "sanda1.pdf"); plt.close(fig)

# Fig. 2: forward feature selection, scored on validation vs on test
S = json.loads((HERE / "out" / "T4b_selection.json").read_text())
fig, axs = plt.subplots(1, 2, figsize=(W * 2, 1.9), sharey=True)
for ax, d in zip(axs, ("QR1", "QR2")):
    for crit, ls, mk in (("test", "-", "o"), ("val", "--", "s")):
        p = S[f"{d}_{crit}"]
        ax.plot(range(1, 8), [x["test_mse"] for x in p], ls=ls, marker=mk, ms=3, color="k", lw=1,
                label=f"scored on {'test' if crit == 'test' else 'validation'}")
        for k, x in enumerate(p):
            ax.annotate(x["added"], (k + 1, x["test_mse"]), fontsize=5.5, xytext=(2, -8 if crit == "test" else 3), textcoords="offset points")
    ax.axhline(mse(P["HAR_te"]), color="k", ls=":", lw=1, label="HAR(1,3,12)")
    ax.set_title(d); ax.set_xlabel("Number of inputs selected"); ax.set_xticks(range(1, 8))
axs[0].set_ylabel("Test MSE (log RV)"); axs[1].legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "sanda2.pdf"); plt.close(fig)

# Fig. 3: commodity transfer, validation-selected reservoir vs HAR
P7 = np.load(HERE / "out" / "preds_T7.npz")
fig, ax = plt.subplots(figsize=(W, 2.1))
for j, (d, off, mk) in enumerate((("QR1", -0.15, "o"), ("QR2", 0.15, "s"))):
    for k, a in enumerate(T.ASSETS):
        _, Sa, ntr, _ = T.build(a); ya = Sa.RV.values; yt, yv = ya[ntr:], ya[ntr - 36:ntr]
        QV = np.array([P7[f"{a}_{d}_{i:03d}_va"] for i in range(100)]); QT = np.array([P7[f"{a}_{d}_{i:03d}_te"] for i in range(100)])
        q = QT[int(np.argmin(((QV - yv) ** 2).mean(1)))]
        r, lo, hi, _ = compare((q - yt) ** 2, (P7[f"{a}_{d}_HAR_te"] - yt) ** 2)
        ax.errorbar(k + off, r, yerr=[[r - lo], [hi - r]], fmt=mk, color="k", mfc="white" if d == "QR2" else "k", ms=3, lw=0.8, capsize=1.5,
                    label=d if k == 0 else None)
ax.axhline(1, color="k", lw=0.6); ax.set_xticks(range(8)); ax.set_xticklabels(T.ASSETS)
ax.set_ylabel("MSE ratio, reservoir / HAR"); ax.set_xlabel("Futures contract (test 2015-2024, 120 months)")
ax.legend(frameon=False, loc="upper left", ncol=2); fig.tight_layout(); fig.savefig(OUT / "sanda3.pdf"); plt.close(fig)

# Fig. 4: power curves (supplementary 1%-step grid), single test
txt = (HERE / "report_T6_fine.txt").read_text().splitlines()
fig, ax = plt.subplots(figsize=(W, 2.1))
styles = iter([("-", "o"), ("--", "s"), ("-.", "^"), (":", "v")])
for i, line in enumerate(txt):
    m = re.match(r"^(Li et al\. design|copper transfer), (\d+) months, (QR[12]):", line)
    if m:
        pw = [(float(a), float(b)) for a, b in re.findall(r"([\d.]+)%:([\d.]+)", txt[i + 1])]
        ls, mk = next(styles)
        ax.plot([a for a, _ in pw], [b for _, b in pw], ls=ls, color="k", lw=0.9, marker=mk, ms=2, markevery=5,
                label=f"{'Li et al. data' if m[1].startswith('Li') else 'copper'}, {m[3]} (n={m[2]})")
ax.axhline(0.8, color="0.5", lw=0.6); ax.set_xlabel("True MSE reduction over HAR (%)"); ax.set_ylabel("Power (single test, 5%)")
ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "sanda4.pdf"); plt.close(fig)

# Fig. 5: hardware vs noiseless 1-step circuit, readout features
H = np.load(HERE / "out" / "T8_full_hw.npz"); T1 = np.load(HERE / "out" / "T8_t1sim.npz")
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.scatter(T1["F"].ravel(), H["F_hw"].ravel(), s=1, color="k", alpha=0.25, lw=0)
ax.plot([-1, 1], [-1, 1], color="0.5", lw=0.6)
ax.set_xlabel(r"Noiseless circuit $\langle Z_q\rangle$"); ax.set_ylabel(r"Hardware $\langle Z_q\rangle$ (500 shots)")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "sanda5.pdf"); plt.close(fig)
print("written to", OUT)
