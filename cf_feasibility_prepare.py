"""Export the *saved* CounterNet Adult representation for the PyTorch comparison.

Run with the existing ReLax environment: .venv/bin/python cf_feasibility_prepare.py
No refitting, row filtering, reshuffling, or change to the original data assets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cf-feasibility-matplotlib")
import numpy as np
from relax.data_module import DataModule

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "trained_models/cf_feasibility_baseline"


def prepare():
    source = ROOT / "relax-assets/adult/data"
    dm = DataModule.load_from_path(source)
    xs, ys = np.asarray(dm.xs, dtype=np.float32), np.asarray(dm.ys).reshape(-1).astype(np.int64)
    train = np.asarray(dm.train_indices, dtype=np.int64)
    test = np.asarray(dm.test_indices, dtype=np.int64)
    assert xs.shape == (32561, 29) and len(train) == 24420 and len(test) == 8141
    assert len(np.intersect1d(train, test)) == 0
    assert np.array_equal(np.sort(np.r_[train, test]), np.arange(len(xs)))
    assert np.isfinite(xs).all() and set(np.unique(ys)) == {0, 1}
    assert np.array_equal(ys, dm.data.income.to_numpy())
    features, encoded_names = [], []
    for f, (start, end) in zip(dm.features, dm.features.feature_indices):
        item = dict(name=f.name, start=int(start), end=int(end), categorical=bool(f.is_categorical))
        tr = f.transformation.transformer
        if f.is_categorical:
            item["categories"] = [str(v) for v in np.asarray(tr.categories_).reshape(-1)]
            assert np.allclose(xs[:, start:end].sum(1), 1)
            encoded_names.extend(f"{f.name}_{v}" for v in item["categories"])
        else:
            item["min"] = float(np.asarray(tr.min_).reshape(-1)[0])
            item["max"] = float(np.asarray(tr.max_).reshape(-1)[0])
            encoded_names.append(f.name)
        features.append(item)
    metadata = dict(
        rows=len(xs), dimensions=xs.shape[1], train_rows=len(train), test_rows=len(test),
        features=features, encoded_names=encoded_names,
        csv_sha256=hashlib.sha256((source / "data.csv").read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        test_indices_sha256=hashlib.sha256(test.tobytes()).hexdigest(),
        encoded_x_sha256=hashlib.sha256(xs.tobytes()).hexdigest(),
        preprocessing="Existing serialized ReLax transforms; original fitting provenance not established here.",
        validation="None: fixed settings and final-epoch checkpoints; no test-based tuning or checkpoint selection.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "adult_inputs.npz", x=xs, y=ys, train_indices=train, test_indices=test)
    (OUT / "data_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ["rows", "dimensions", "train_rows", "test_rows", "encoded_x_sha256"]}))


if __name__ == "__main__":
    prepare()
