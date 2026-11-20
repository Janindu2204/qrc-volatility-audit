"""Test 8 reference T1SIM: the same 1-step Trotter circuit as t8_submit.py on the noiseless Aer density-matrix simulator, all rows 3..815.
Run in .venv-qiskit: python t8_t1sim.py. Output: out/T8_t1sim.npz."""
import sys
from pathlib import Path
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.argv = sys.argv[:1] + ["full"]
import t8_submit as S

D = np.load(S.DATA / "t8_inputs.npz"); rows = np.arange(3, 816)
dm = AerSimulator(method="density_matrix")
base = S.build(D["J"]).remove_final_measurements(inplace=False); base.save_density_matrix()
tb = transpile(base, dm)
b = np.arange(2 ** S.NQ); zs = 1 - 2 * ((b[:, None] >> np.arange(S.NQ)) & 1)
F = np.zeros((len(rows), S.NQ))
for k, p in enumerate(S.params(D["X"], rows)):
    rho = dm.run(tb.assign_parameters(p)).result().data(0)["density_matrix"]
    F[k] = np.real(np.diag(np.asarray(rho))) @ zs
    if (k + 1) % 100 == 0:
        print(f"{k + 1}/{len(rows)}", flush=True)
np.savez(S.OUT / "T8_t1sim.npz", rows=rows, F=F)
print("saved")
