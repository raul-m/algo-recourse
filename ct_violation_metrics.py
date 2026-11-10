"""Per-feature constraint violations for the CT runs on the CounterNet split.

Same definitions as the CounterNet evaluation (thesis_immutability_experiment.py,
cred_grid.py), applied to CT's counterfactuals after snapping every one-hot block
to its argmax: fraction of test rows whose fixed feature changed, whose age
decreased, or whose counterfactual lies in a banned category. Read-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ct_data"
RUNS = ROOT / "ct_runs_cn"
EVAL = __import__("os").environ.get("EVAL_SUBDIR", "eval_relaxed")

SPEC = {
    "adult": dict(fixed=["race", "gender", "marital_status"], age="age",
                  banned={"workclass": ["Other/Unknown"], "occupation": ["Other/Unknown"]}),
    "credit": dict(fixed=["SEX", "MARRIAGE"], age="AGE", banned={"EDUCATION": ["0", "5", "6"]}),
    "oulad": dict(fixed=["gender", "region", "disability"], age="age_band",
                  banned={"imd_band": ["0"]}),
}


def violations(src, run_dir):
    meta = json.loads((DATA / src / "meta.json").read_text())
    spec = SPEC["credit" if src.startswith("credit") else src]
    X = np.loadtxt(DATA / src / "X_test.csv", delimiter=",")
    spans = {f["name"]: (f["start"], f["end"]) for f in meta["features"]}
    cols = meta["columns"]
    out = {}
    for exp, label in [("experiment_1", "CT"), ("experiment_2", "baseline")]:
        CF = np.loadtxt(run_dir / EVAL / f"{exp}_counterfactuals.csv", delimiter=",")
        CF = np.atleast_2d(CF)
        for f in meta["features"]:            # snap one-hot blocks to argmax
            if f["categorical"]:
                s, e = f["start"], f["end"]
                k = CF[:, s:e].argmax(1)
                CF[:, s:e] = 0
                CF[np.arange(len(CF)), s + k] = 1
        r, any_ch = {}, np.zeros(len(X), bool)
        for name in spec["fixed"]:
            s, e = spans[name]
            ch = np.abs(CF[:, s:e] - X[:, s:e]).sum(1) > 1e-6
            r[f"viol_{name}"] = ch.mean()
            any_ch |= ch
        r["viol_any"] = any_ch.mean()
        s, e = spans[spec["age"]]
        if e - s > 1:   # ordinal one-hot (OULAD age_band): band index decreased
            r["age_decrease"] = (CF[:, s:e].argmax(1) < X[:, s:e].argmax(1)).mean()
        else:
            r["age_decrease"] = (CF[:, s:e] < X[:, s:e] - 1e-6).any(1).mean()
        dis = np.zeros(len(X), bool)
        dis_factual = np.zeros(len(X), bool)
        for name, labels in spec["banned"].items():
            for lab in labels:
                j = cols.index(f"{name}__{lab}")
                dis |= CF[:, j] > 0.5
                dis_factual |= X[:, j] > 0.5
        r["banned_category"] = dis.mean()
        r["banned_category_already_in_factual"] = (dis & dis_factual).mean()
        out[label] = {k: float(v) for k, v in r.items()}
    return out


if __name__ == "__main__":
    results = {}
    for run in sorted(p for p in RUNS.iterdir() if (p.is_dir() and p.name.startswith("cn_"))):
        for d in run.glob("*/mlp/*"):
            if (d / EVAL / "experiment_1_counterfactuals.csv").exists():
                src = {"cn_adult": "adult", "cn_credit": "credit",
                       "cn_credit_balanced": "credit_balanced", "cn_oulad": "oulad"}[d.name]
                results[run.name] = violations(src, d)
                (d / EVAL / "violations.json").write_text(
                    json.dumps(results[run.name], indent=2))
    for k, v in results.items():
        for lab, r in v.items():
            print(f"{k:34s} {lab:8s} " + "  ".join(f"{m}={x:.3f}" for m, x in r.items()))
