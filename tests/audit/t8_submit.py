"""Test 8 (prereg_li_audit_T8.json): submit the 1-step Trotter QR1 circuit to IBM hardware and save hardware features.
Run in .venv-qiskit: python t8_submit.py pilot|full. Job ids go to out/T8_jobs.json as soon as each job is created."""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE.parents[1] / "data" / "li2026"
NQ, NIN, K = 10, 7, 3
MODE = sys.argv[1]
SHOTS = {"pilot": 1000, "full": 500}[MODE]
ROWS = np.linspace(3, 815, 60).round().astype(int) if MODE == "pilot" else np.arange(3, 816)


def build(J):
    H = SparsePauliOp.from_sparse_list([("XX", [i, j], J[i, j]) for i in range(NQ) for j in range(i + 1, NQ) if J[i, j] != 0]
                                       + [("Z", [q], 1.0) for q in range(NQ)], NQ)
    th = ParameterVector("x", K * NIN)
    qc = QuantumCircuit(NQ, NQ)
    for kk, k in enumerate(range(K, 0, -1)):
        if k < K:
            for q in range(NIN):
                qc.reset(q)
        for q in range(NIN):
            qc.ry(np.pi * th[kk * NIN + q], q)
        qc.append(PauliEvolutionGate(H, time=1.0, synthesis=SuzukiTrotter(order=2, reps=1)), range(NQ))
    qc.measure(range(NQ), range(NQ))
    return qc


def params(X, rows):
    return np.array([np.concatenate([X[r - k] for k in range(K, 0, -1)]) for r in rows])


def z_from_counts(counts, shots):
    z = np.zeros(NQ)
    for bits, c in counts.items():
        b = int(bits, 2)
        z += c * (1 - 2 * ((b >> np.arange(NQ)) & 1))
    return z / shots


if __name__ == "__main__":
    h = hashlib.sha256((HERE / "prereg_li_audit_T8.json").read_bytes()).hexdigest()
    assert h == (HERE / "prereg_li_audit_T8.sha256").read_text().split()[0], "registration changed after freezing"
    D = np.load(DATA / "t8_inputs.npz"); X = D["X"]
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler import generate_preset_pass_manager
    svc = QiskitRuntimeService(channel="ibm_cloud")
    cands = [svc.backend(n) for n in ("ibm_fez", "ibm_marrakesh", "ibm_kingston")]
    be = min((b for b in cands if b.status().operational), key=lambda b: b.status().pending_jobs)
    isa = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=1).run(build(D["J"]))
    print(f"{MODE}: backend {be.name} (queue {be.status().pending_jobs}), {len(ROWS)} circuits x {SHOTS} shots, "
          f"CZ {isa.count_ops().get('cz', 0)}, depth {isa.depth()}", flush=True)
    job = SamplerV2(mode=be).run([(isa, params(X, ROWS))], shots=SHOTS)
    log = json.loads((OUT / "T8_jobs.json").read_text()) if (OUT / "T8_jobs.json").exists() else {}
    log[MODE] = {"job_id": job.job_id(), "backend": be.name, "shots": SHOTS, "n_circuits": len(ROWS), "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (OUT / "T8_jobs.json").write_text(json.dumps(log, indent=1)); print("job", job.job_id(), flush=True)
    res = job.result()
    bits = res[0].data.c
    F = np.array([z_from_counts(bits.get_counts(i), SHOTS) for i in range(len(ROWS))])
    usage = job.usage() if hasattr(job, "usage") else None
    log[MODE].update({"status": str(job.status()), "usage_seconds": usage, "metrics": job.metrics().get("usage", {})})
    (OUT / "T8_jobs.json").write_text(json.dumps(log, indent=1, default=str))
    np.savez(OUT / f"T8_{MODE}_hw.npz", rows=ROWS, F_hw=F, shots=SHOTS)
    print(json.dumps(log[MODE], indent=1, default=str))
