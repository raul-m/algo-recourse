"""Model-approx per-seed metrics in the results page's columns, from the saved test candidates.

For each dataset and each of the page's four Model-approx rows (C1 and C2, native and with the
feasibility set at alpha = 1) and each generator seed (42, 43, 44): accuracy, CounterNet validity,
recourse validity, proximity, any-validity recourse cost, recourse cost, the number of changed
features, and the shares of counterfactuals changing an immutable feature and breaking any rule
of the feasibility set, over all 10 draws per person. Read-only with respect to all trained models; writes
utils/stats/model_approx_seed42_metrics.json.
Needs torch (the Model-approx environment):

    .venv-cf-feasibility/bin/python utils/model_approx_seed42_metrics.py
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "utils/stats/model_approx_seed42_metrics.json"
sys.path.insert(0, str(ROOT))
import cf_feasibility_experiment as base
import cf_feasibility_c2_experiment as c2
import cf_feasibility_multidata as md

FAV = {"adult": 1, "credit": 0, "oulad": 1}
TH = 1e-2
ADULT_POLICY = dict(fixed=["race", "gender", "marital_status"], age="age", age_kind="continuous",
                    banned={"workclass": ["Other/Unknown"], "occupation": ["Other/Unknown"]})


def runs():
    """(dataset, row, seed) -> (x, y, classifier, features, candidates path)."""
    out = {}
    for setting, old, cfg in (("c1", base.DEFAULT_OUT, base.Config()), ("c2", c2.OUT, c2.C2Config())):
        a, meta = base.load_data(old)
        clf = base.frozen_classifier(old, cfg)
        x = a["x"][a["test_indices"]]; y = a["y"][a["test_indices"]].reshape(-1)
        for variant, directory, row in (("original", old, f"{setting} native"),
                                        ("loss_only", ROOT / "trained_models/cf_feasibility_penalties" / setting, f"{setting} a1")):
            for seed in (42, 43, 44):
                out[("adult", row, seed)] = (x, y, clf, meta["features"], directory / f"{variant}_seed{seed}/test_candidates.npz",
                                             ADULT_POLICY)
    for ds in ("credit", "oulad"):
        o = md.OUT / ds
        a, meta = md.load_data(o)
        clf = md.frozen_classifier(o, base.Config())
        x = a["x"][a["test_indices"]]; y = a["y"][a["test_indices"]].reshape(-1)
        for setting in ("c1", "c2"):
            for alpha, tag in ((0.0, "native"), (1.0, "a1")):
                for seed in (42, 43, 44):
                    out[(ds, f"{setting} {tag}", seed)] = (x, y, clf, meta["features"],
                                                           md.run_path(o, alpha, seed, setting) / "test_candidates.npz",
                                                           meta["policy"])
    return out


def any_immutable(cf, x, features, policy):
    """Per draw and row: changes at least one immutable feature."""
    by = {f["name"]: f for f in features}
    flag = np.zeros(cf.shape[:2], bool)
    for name in policy["fixed"]:
        s, e = by[name]["start"], by[name]["end"]
        flag |= cf[..., s:e].argmax(-1) != x[None, :, s:e].argmax(-1)
    return flag


def any_rule(cf, x, features, policy):
    """Per draw and row: breaks an immutable feature, lowers age, or ends in an excluded category."""
    by = {f["name"]: f for f in features}
    flag = any_immutable(cf, x, features, policy)
    s, e = by[policy["age"]]["start"], by[policy["age"]]["end"]
    flag |= (cf[..., s] < x[None, :, s] - 1e-6) if policy["age_kind"] == "continuous" else         (cf[..., s:e].argmax(-1) < x[None, :, s:e].argmax(-1))
    for name, cats in policy["banned"].items():
        f = by[name]
        idx = [f["start"] + f["categories"].index(c) for c in cats]
        flag |= (cf[..., idx] > .5).any(-1)
    return float(flag.mean())


def metrics(ds, x, y, clf, features, path, policy):
    with np.load(path) as z:
        cf = z["cf_decoded"]
    with torch.no_grad():
        fx = clf(torch.from_numpy(x)).argmax(-1).numpy()
        fc = clf(torch.from_numpy(cf)).argmax(-1).numpy()
    d = np.abs(cf - x[None]).sum(-1)
    rej = fx != FAV[ds]
    ok = rej[None] & (fc != fx[None])
    n = np.zeros(d.shape)
    for f in features:
        s, e = f["start"], f["end"]
        if f.get("categories") and e - s > 1:
            n += cf[..., s:e].argmax(-1) != x[None, :, s:e].argmax(-1)
        else:
            n += np.abs(cf[..., s] - x[None, :, s]) > TH
    return dict(acc=float((fx == y).mean()), validity=float((fc != fx[None]).mean()),
                recourse_validity=float(ok.sum() / (rej.sum() * cf.shape[0])),
                proximity=float(d.mean()), any_cost=float(d[:, rej].mean()),
                recourse_cost=float(d[ok].mean()), n_changes=float(n[ok].mean()),
                any_rule=any_rule(cf, x, features, policy),
                any_immutable=float(any_immutable(cf, x, features, policy).mean()))


if __name__ == "__main__":
    torch.set_num_threads(4)
    res = {}
    for (ds, row, seed), (x, y, clf, feats, path, policy) in runs().items():
        res[f"{ds}|{row}|{seed}"] = metrics(ds, x, y, clf, feats, path, policy)
        print(ds, row, seed, {k: round(v, 4) for k, v in res[f"{ds}|{row}|{seed}"].items()}, flush=True)
    OUT.write_text(json.dumps(res, indent=1))
