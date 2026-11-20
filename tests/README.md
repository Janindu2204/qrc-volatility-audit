# Tests

All code is in `tests/audit/`. Each test was registered before it was run; the registration and its SHA-256 are next to the code.

| Test | Registration | Run | Report |
|---|---|---|---|
| T1-T3 reproduction, honest selection, controls | `prereg_li_audit_T2T3.json` (+ addendum `addendum_A1_controls_2026-11-13.json`) | `t2t3_run.py`, `ctrlfix_run.py` | `t2t3_report.py` |
| T4-T6 selection, trigonometric description, power | `prereg_li_audit_T4T6.json` | `t4_run.py`, `t5_run.py`, `t6_run.py` | `t4a_report.py` |
| T7 transfer to commodity futures | `prereg_li_audit_T7.json` | `t7_run.py` | `t7_report.py` |
| T8 IBM quantum hardware | `prereg_li_audit_T8.json` | `t8_plan.py`, `t8_submit.py`, `t8_t1sim.py` | `t8_score.py` |

Shared code: `li_lib.py` (independent implementation of the audited reservoir), `stats_lib.py` (tests and intervals).
Figures: `python make_figures.py <output folder>`.
T8 needs an IBM Quantum account saved locally (`QiskitRuntimeService(channel="ibm_cloud")`); no credential is stored in this repository.
