"""Export the CounterNet Adult, Credit and OULAD data for Counterfactual Training (CT).

Read-only with respect to the CounterNet experiments: the data are loaded through
``relax.load_data`` exactly as the CounterNet runs load them, which reuses the
train/test indices stored in ``relax-assets/<name>/data/config.json``. Nothing is
written back. Outputs go to ``../ct_data/<name>/`` (or ``$CT_DATA_OUT/<name>/``):

    X_train.csv, y_train.csv, X_test.csv, y_test.csv   one row per instance, no header
    meta.json                                          columns, one-hot groups, and the
                                                       CT mutability vectors

Mutability vectors (CT's own mechanism, one entry per encoded column). The reported
results use arms A, B and C; each run also stores its vector in its grid_config.toml.

    arm A  "ct_original"        CT's published constraint only: age frozen
    arm B  "ct_extended_dec"    age frozen, CounterNet's fixed features frozen, banned
                                categories decrease-only (can be left, never entered)
    arm C  "ct_counternet_dec"  as B, but age non-decreasing (CounterNet's treatment)

Adult and Credit also keep two superseded vectors from the first runs, in which the
banned categories were frozen instead of decrease-only: "ct_extended" and
"ct_counternet". They are written so the earlier run directories stay traceable;
``frozen_columns`` lists the frozen columns of arm A and these two vectors only.
OULAD was first run after the switch to decrease-only and has no superseded vectors.
"""
import json
import os
from pathlib import Path

import numpy as np
import relax

from thesis_immutability_experiment import (
    AGE as ADULT_AGE,
    DISALLOWED_MUTABLE_CATEGORIES as ADULT_BANNED,
    IMMUTABLE as ADULT_FIXED,
    feature_col_mask,
)

OUT_ROOT = Path(os.environ.get("CT_DATA_OUT",
                               Path(__file__).resolve().parent.parent / "ct_data"))

# Credit rules as used in cred_grid.py (copied, not imported, to avoid running it).
CREDIT_FIXED = ["SEX", "MARRIAGE"]
CREDIT_AGE = "AGE"
CREDIT_BANNED = {"EDUCATION": [0, 5, 6]}

# OULAD rules as used in oulad_grid.py. age_band is an ascending ordinal one-hot.
OULAD_FIXED = ["gender", "region", "disability"]
OULAD_AGE = "age_band"
OULAD_BANNED = {"imd_band": ["0"]}

SPECS = {
    "adult": dict(age=ADULT_AGE, fixed=ADULT_FIXED, banned=ADULT_BANNED),
    "credit": dict(age=CREDIT_AGE, fixed=CREDIT_FIXED, banned=CREDIT_BANNED),
}


def export_oulad():
    """OULAD with three arms. CT never ran OULAD, so arm A mirrors CT's choice on Adult
    and Credit (age frozen). Arm C carries the CounterNet rules: gender, region and
    disability fixed, imd_band '0' banned (decrease-only), and age_band non-decreasing,
    approximated per column: youngest band decrease-only (can be left, never entered),
    middle unconstrained, oldest increase-only (can be entered, never left). Arm B is
    arm C with age_band frozen, as on Adult and Credit."""
    name = "oulad"
    dm = relax.load_data(name)
    xs, ys = np.asarray(dm.xs, dtype=np.float32), np.asarray(dm.ys).ravel().astype(int)
    tr, te = np.asarray(dm.train_indices), np.asarray(dm.test_indices)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(xs)
    names, spans = column_layout(dm)
    by = {s["name"]: s for s in spans}
    age = list(range(by[OULAD_AGE]["start"], by[OULAD_AGE]["end"]))
    assert [names[j] for j in age] == ["age_band__0-35", "age_band__35-55", "age_band__55<="]
    fixed = np.asarray(feature_col_mask(dm, OULAD_FIXED)).astype(bool)
    banned = banned_mask(dm, OULAD_BANNED)
    a = np.full(xs.shape[1], "both", dtype=object); a[age] = "none"
    c = np.full(xs.shape[1], "both", dtype=object)
    c[fixed] = "none"; c[banned] = "decrease"
    c[age[0]], c[age[-1]] = "decrease", "increase"
    b = np.full(xs.shape[1], "both", dtype=object)
    b[fixed] = "none"; b[banned] = "decrease"; b[age] = "none"
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "X_train.csv", xs[tr], delimiter=",", fmt="%.8g")
    np.savetxt(out / "X_test.csv", xs[te], delimiter=",", fmt="%.8g")
    np.savetxt(out / "y_train.csv", ys[tr], fmt="%d")
    np.savetxt(out / "y_test.csv", ys[te], fmt="%d")
    meta = dict(
        dataset=name,
        source="relax.load_data (CounterNet experiments), stored train/test indices",
        n_train=int(len(tr)), n_test=int(len(te)), n_columns=int(xs.shape[1]),
        columns=names, features=spans,
        categorical_groups=[list(range(s["start"] + 1, s["end"] + 1))
                            for s in spans if s["categorical"]],
        continuous=[s["start"] + 1 for s in spans if not s["categorical"]],
        mutability=dict(ct_original=a.tolist(), ct_counternet_dec=c.tolist(),
                        ct_extended_dec=b.tolist()),
        label_counts=dict(train=np.bincount(ys[tr]).tolist(), test=np.bincount(ys[te]).tolist()),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for arm, v in meta["mutability"].items():
        summary = {k: [names[j] for j, x in enumerate(v) if x == k] for k in ("none", "increase", "decrease")}
        print(f"oulad {arm}: none {len(summary['none'])} cols, increase {summary['increase']}, decrease {summary['decrease']}")


def column_layout(dm):
    """Per-column names plus feature spans, using the data module's own indices."""
    names, spans = [], []
    for feat, (s, e) in zip(dm.features, dm.features.feature_indices):
        if feat.is_categorical:
            cats = [str(c) for c in
                    np.asarray(feat.transformation.transformer.categories_).ravel()]
            assert len(cats) == e - s
            names += [f"{feat.name}__{c}" for c in cats]
        else:
            assert e - s == 1
            names.append(feat.name)
        spans.append(dict(name=feat.name, start=int(s), end=int(e),
                          categorical=bool(feat.is_categorical)))
    return names, spans


def banned_mask(dm, banned):
    mask = np.zeros(dm.xs.shape[1], dtype=bool)
    for feat, (s, e) in zip(dm.features, dm.features.feature_indices):
        for label in banned.get(feat.name, []):
            cats = [str(c) for c in
                    np.asarray(feat.transformation.transformer.categories_).ravel()]
            mask[s + cats.index(str(label))] = True
    return mask


def export(name):
    spec = SPECS[name]
    dm = relax.load_data(name)
    xs, ys = np.asarray(dm.xs, dtype=np.float32), np.asarray(dm.ys).ravel().astype(int)
    tr, te = np.asarray(dm.train_indices), np.asarray(dm.test_indices)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(xs)

    names, spans = column_layout(dm)
    d = xs.shape[1]

    age = np.asarray(feature_col_mask(dm, [spec["age"]])).astype(bool)
    fixed = np.asarray(feature_col_mask(dm, spec["fixed"])).astype(bool)
    banned = banned_mask(dm, spec["banned"])
    assert age.sum() == 1

    arm_a = np.where(age, "none", "both")
    # Reported arms B and C: banned categories decrease-only.
    arm_b = np.where(age | fixed, "none", np.where(banned, "decrease", "both"))
    arm_c = np.where(fixed, "none", np.where(banned, "decrease",
                                             np.where(age, "increase", "both")))
    # Superseded first runs: banned categories frozen.
    old_b = np.where(age | fixed | banned, "none", "both")
    old_c = np.where(fixed | banned, "none", np.where(age, "increase", "both"))

    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    fmt = "%.8g"
    np.savetxt(out / "X_train.csv", xs[tr], delimiter=",", fmt=fmt)
    np.savetxt(out / "X_test.csv", xs[te], delimiter=",", fmt=fmt)
    np.savetxt(out / "y_train.csv", ys[tr], fmt="%d")
    np.savetxt(out / "y_test.csv", ys[te], fmt="%d")

    meta = dict(
        dataset=name,
        source="relax.load_data (CounterNet experiments), stored train/test indices",
        n_train=int(len(tr)), n_test=int(len(te)), n_columns=int(d),
        columns=names,
        features=spans,
        # 1-indexed for Julia
        categorical_groups=[list(range(s["start"] + 1, s["end"] + 1))
                            for s in spans if s["categorical"]],
        continuous=[s["start"] + 1 for s in spans if not s["categorical"]],
        mutability=dict(ct_original=arm_a.tolist(), ct_extended=old_b.tolist(),
                        ct_counternet=old_c.tolist(), ct_extended_dec=arm_b.tolist(),
                        ct_counternet_dec=arm_c.tolist()),
        frozen_columns=dict(
            ct_original=[names[i] for i in np.flatnonzero(arm_a == "none")],
            ct_extended=[names[i] for i in np.flatnonzero(old_b == "none")],
            ct_counternet=[names[i] for i in np.flatnonzero(old_c == "none")],
        ),
        label_counts=dict(train=np.bincount(ys[tr]).tolist(),
                          test=np.bincount(ys[te]).tolist()),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: X {xs.shape}, train {len(tr)}, test {len(te)}")
    for arm, v in meta["mutability"].items():
        summary = {k: [names[j] for j, x in enumerate(v) if x == k] for k in ("increase", "decrease")}
        print(f"  {arm}: none {v.count('none')} cols, increase {summary['increase']}, "
              f"decrease {summary['decrease']}")


def export_balanced_train(name, seed=2025):
    """Copy of ``ct_data/<name>`` whose TRAINING rows are class-balanced by dropping
    majority-class rows (as TaijaData balances CT's own Credit data). The test rows
    are CounterNet's, unchanged. Written to ``ct_data/<name>_balanced``."""
    import shutil
    src, dst = OUT_ROOT / name, OUT_ROOT / f"{name}_balanced"
    dst.mkdir(parents=True, exist_ok=True)
    X = np.loadtxt(src / "X_train.csv", delimiter=",", dtype=np.float32)
    y = np.loadtxt(src / "y_train.csv", dtype=int)
    rng = np.random.default_rng(seed)
    n_min = np.bincount(y).min()
    keep = np.sort(np.concatenate(
        [rng.choice(np.flatnonzero(y == k), n_min, replace=False) for k in (0, 1)]))
    np.savetxt(dst / "X_train.csv", X[keep], delimiter=",", fmt="%.8g")
    np.savetxt(dst / "y_train.csv", y[keep], fmt="%d")
    for f in ("X_test.csv", "y_test.csv"):
        shutil.copyfile(src / f, dst / f)
    meta = json.loads((src / "meta.json").read_text())
    meta.update(dataset=f"{name}_balanced", n_train=int(len(keep)),
                label_counts=dict(train=np.bincount(y[keep]).tolist(),
                                  test=meta["label_counts"]["test"]),
                balancing=f"training rows undersampled to {n_min} per class, seed {seed}; "
                          "test rows unchanged")
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}_balanced: train {len(keep)} {np.bincount(y[keep]).tolist()}, test unchanged")


if __name__ == "__main__":
    import sys
    if "--balanced" in sys.argv:
        export_balanced_train("credit")
    elif "--oulad" in sys.argv:
        export_oulad()
    else:
        for n in SPECS:
            export(n)
