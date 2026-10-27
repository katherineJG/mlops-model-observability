"""Versioned reference dataset with provenance.

The reference is the healthy baseline that current windows are compared against: the 10 model
inputs, the model's churn probability and the true label, for 5000 rows from the seeded
generator. Run `python -m monitoring.reference` to rebuild it; the output is deterministic.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from app.model_loader import load_model
from monitoring.synthetic import generate, score

VERSION = "reference-v1"
SEED = 42
N_ROWS = 5000


def build_reference(out_dir: Path, model_dir: Path) -> Path:
    """Write `<VERSION>.csv` and `<VERSION>.json` to out_dir. Returns the CSV path."""
    model = load_model(model_dir)
    frame = generate(N_ROWS, SEED)
    frame["churn_probability"] = score(model, frame).round(6)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{VERSION}.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    provenance = {
        "version": VERSION,
        "rows": len(frame),
        "columns": list(frame.columns),
        "generator": "monitoring.synthetic.generate (copy of Project 01 generate_synthetic)",
        "seed": SEED,
        "n_rows": N_ROWS,
        "equals": "Project 01 data/raw/raw.csv feature and churn columns, row for row "
        "(compared when this reference was created; customer_id is deliberately not kept)",
        "model_id": model.manifest.model_id,
        "model_version": model.manifest.model_version,
        "model_sha256": model.manifest.sha256,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "synthetic": True,
    }
    (out_dir / f"{VERSION}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return csv_path


def load_reference(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the reference after checking its CSV against the recorded SHA-256."""
    provenance = json.loads((directory / f"{VERSION}.json").read_text())
    data = (directory / f"{VERSION}.csv").read_bytes()
    if hashlib.sha256(data).hexdigest() != provenance["csv_sha256"]:
        raise ValueError(f"{VERSION}.csv does not match its recorded checksum")
    return pd.read_csv(directory / f"{VERSION}.csv"), provenance


if __name__ == "__main__":
    print(build_reference(Path("reference"), Path("models/churn-v1")))
