"""Test 8 planning only: nothing is submitted. Builds Li et al.'s QR1 reservoir as a gate circuit (mid-circuit reset of the 7 input
qubits per lag, PauliEvolutionGate with 2nd-order Suzuki-Trotter), checks Trotter error against the exact li_lib features on a noiseless
density-matrix simulator, transpiles for an IBM backend (read-only backend query) and estimates QPU time with IBM's planning formula.
Run in .venv-qiskit: python t8_plan.py <ref.npz from li_lib> [backend]."""
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

R = np.load(sys.argv[1]); BACKEND = sys.argv[2] if len(sys.argv) > 2 else "ibm_fez"
J, X, idx, Fref = R["J"][50], R["X"], R["idx"], R["F"]          # QR1, validation-selected instance 51
NQ, NIN, K = 10, 7, 3
H = SparsePauliOp.from_sparse_list([("XX", [i, j], J[i, j]) for i in range(NQ) for j in range(i + 1, NQ) if J[i, j] != 0]
                                   + [("Z", [q], 1.0) for q in range(NQ)], NQ)


def circuit(row, reps, measure=False):
    qc = QuantumCircuit(NQ, NQ if measure else 0)
    for k in range(K, 0, -1):
        if k < K:
            for q in range(NIN):
                qc.reset(q)
        for q in range(NIN):
            qc.ry(np.pi * X[row - k, q], q)
        qc.append(PauliEvolutionGate(H, time=1.0, synthesis=SuzukiTrotter(order=2, reps=reps)), range(NQ))
    if measure:
        qc.measure(range(NQ), range(NQ))
    else:
        qc.save_density_matrix()
    return qc


def z_expect(rho):
    p = np.real(np.diag(np.asarray(rho)))
    b = np.arange(2 ** NQ)
    return np.array([p @ (1 - 2 * ((b >> q) & 1)) for q in range(NQ)])


sim = AerSimulator(method="density_matrix")
print(f"QR1 instance 51, {len(idx)} check rows; exact reference from li_lib\n")
print("Trotter reps | max |<Z> error| | median |<Z> error|")
for reps in (1, 2, 4, 8):
    circs = [transpile(circuit(r, reps), sim) for r in idx]
    res = sim.run(circs).result()
    F = np.array([z_expect(res.data(i)["density_matrix"]) for i in range(len(idx))])
    e = np.abs(F - Fref)
    print(f"{reps:12d} | {e.max():15.4f} | {np.median(e):18.4f}")

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
be = QiskitRuntimeService(channel="ibm_cloud").backend(BACKEND)
pm = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=1)
tg = be.target
twoq = next(g for g in ("cz", "ecr", "cx") if g in tg.operation_names)
err2 = np.median([p.error for p in tg[twoq].values() if p is not None and p.error is not None])
print(f"\nBackend {BACKEND}: native 2-qubit gate {twoq}, median error {err2:.2e}")
print("Trotter reps | 2q gates | depth | resets | rough success estimate (1-e)^n2q")
for reps in (1, 2, 4):
    t = pm.run(circuit(int(idx[5]), reps, measure=True))
    ops = t.count_ops(); n2 = ops.get(twoq, 0)
    print(f"{reps:12d} | {n2:8d} | {t.depth():5d} | {ops.get('reset', 0):6d} | {(1 - err2) ** n2:.2e}")

print("\nQPU time (IBM planning formula ~ 2 s + 0.00035 s x circuits x shots per job)")
for nc, lbl in ((813, "all rows (hardware-trained readout)"), (245, "test rows only"), (60, "60-row pilot")):
    for shots in (500, 1000):
        print(f"  {lbl:38s} {nc:4d} circuits x {shots:4d} shots: ~{2 + 0.00035 * nc * shots:6.0f} s")
