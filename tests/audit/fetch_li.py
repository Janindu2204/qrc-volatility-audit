"""Download the public data of Li et al. (Phys. Rev. Research 8, 023028, 2026) at a pinned commit into data/li2026/ and record sha256.
The source repository has no licence file: keep these files local (not committed) and use them only to reproduce and audit the paper."""
import hashlib, json, urllib.request
from pathlib import Path

REPO = "LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting"
COMMIT = "d2e9b0aa599afd33c6cfb10352e2c36fe92fa76f"
FILES = ["Data.CSV", "coeff_10.jld2", "predict_result.csv"]
OUT = Path(__file__).resolve().parents[2] / "data" / "li2026"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitignore").write_text("*\n")
    rec = {"repo": REPO, "commit": COMMIT, "files": {}}
    for f in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{f}"
        b = urllib.request.urlopen(url, timeout=60).read()
        (OUT / f).write_bytes(b)
        rec["files"][f] = hashlib.sha256(b).hexdigest()
        print(f, len(b), rec["files"][f])
    (OUT / "SOURCE.json").write_text(json.dumps(rec, indent=1))
