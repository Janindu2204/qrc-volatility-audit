# Audit of a quantum reservoir computing claim for realized-volatility forecasting

Code, registrations, results and manuscript for the paper:

> S. J. I. Sandanayaka, "Does a Quantum Reservoir Computing Advantage in Volatility Forecasting Survive Honest Evaluation? A
> Pre-Registered Audit" (manuscript, 2026).

The audited study is Q. Li, C. Mukhopadhyay, A. Bayat, A. Habibnia, "Quantum reservoir computing for realized volatility
forecasting", Phys. Rev. Research 8, 023028 (2026), doi:10.1103/rbj7-4wnq.

## Folders

| Folder | What it holds |
|---|---|
| `paper/` | The manuscript (`main.tex`, `main.docx`), figures (`figs/`) and the IEEE Access template files |
| `tests/audit/` | All analysis code, the pre-registrations (`prereg_li_audit_*.json`) and the addendum, each with its SHA-256 file |
| `data/` | No data files. Where every data set comes from, and the scripts that rebuild them |
| `results/` | The outputs used in the paper: reports (`reports/`), run logs (`logs/`) and saved predictions (`outputs/`) |

## Check the registrations

    cd tests/audit
    for f in *.sha256; do sha256sum -c "$f"; done

Every registration file in `tests/audit/` must report `OK`. The code is byte-identical to the code used for the paper.

## Reproduce

1. `pip install -r requirements.txt` (Python 3.12).
2. Rebuild the data: see `data/README.md`.
3. Run the tests: see `tests/README.md`. A fresh run writes to `tests/audit/out/`; the archived outputs are in `results/`.

## Data licences

No third-party data are redistributed here. The data of Li et al. are downloaded from their public repository at a pinned commit;
commodity, VIX and dollar-index series are downloaded from the Yahoo Finance chart interface, whose terms do not clearly allow
redistribution. The saved predictions in `results/outputs/` contain model forecasts and the realized-variance targets derived from
those series.
