"""Results-page statistics for CT (Counterfactual Training), arms A, B and C, under CT's original
evaluation, from the outputs the CT harness (ct_harness/counternet_style_eval.jl) saved for each
of the ten search settings of that evaluation. Read-only.

    PYTHONPATH=. .venv/bin/python utils/results_page_ct_stats.py      (from the jax-relax root)

Arms: A, CT's own constraint (age frozen); B, the feasibility set with age frozen; C, the
feasibility set. Every column is computed per search setting and averaged over the ten settings,
as CT's evaluation does; the CF generation time is the median over the settings. Counterfactuals
are snapped to one category per one-hot block, as in results_model_stats.py. Writes
utils/stats/results_page_ct_stats.json.
"""
import json, statistics, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_model_stats as R

ROOT = R.ROOT
OUT = ROOT / "utils/stats/results_page_ct_stats.json"
CT_RUNS, CT_DATA = R.CT_RUNS, R.CT_DATA
ARMS = {"adult": ("cn_adult", {"A": "cn_adult_ct_original/ct_original",
                               "B": "cn_adult_ct_extended_dec/ct_extended",
                               "C": "cn_adult_ct_counternet_dec/ct_counternet"}),
        "credit": ("cn_credit", {"A": "cn_credit_ct_original/ct_original",
                                 "B": "cn_credit_ct_extended_dec/ct_extended",
                                 "C": "cn_credit_ct_counternet_dec/ct_counternet"}),
        "oulad": ("cn_oulad", {"A": "cn_oulad_ct_original/ct_original",
                               "B": "cn_oulad_ct_extended/ct_extended",
                               "C": "cn_oulad_ct_counternet/ct_counternet"})}
# CT's evaluation grid: five energy weights, each with cost weight 0.001 and 0
SETTINGS = (["eval_snapped"] + [f"eval_snapped_egy{e}" for e in ("0.1", "0.5", "1.0", "10.0")]
            + [f"eval_snapped_cost0_egy{e}" for e in ("0.1", "0.5", "1.0", "5.0", "10.0")])
RULES = {"adult": (R.T.IMMUTABLE, R.T.AGE, R.T.DISALLOWED_MUTABLE_CATEGORIES),
         "credit": (R.C.FIXED, R.C.AGE, R.C.BANNED),
         "oulad": (R.O.FIXED, R.O.AGE, R.O.BANNED)}
TH = 1e-2  # a continuous feature counts as changed above 1% of its range


def nchg(X, CF, blk):
    n = np.zeros(len(X))
    for _, cat, s, e in blk:
        n += (CF[:, s:e].argmax(1) != X[:, s:e].argmax(1)) if cat else (np.abs(CF[:, s] - X[:, s]) > TH)
    return n


def rule_shares(ds, X, CF, dm):
    """Shares changing each immutable feature, lowering age (OULAD: a lower age band), ending in
    an excluded category, and breaking any of these rules, as the datasets' evaluators count them."""
    fixed, age, banned = RULES[ds]
    idx = {f.name: r for f, r in zip(dm.features, dm.features.feature_indices)}
    out, br = {}, np.zeros(len(X), bool)
    for name in fixed:
        s, e = idx[name]
        ch = np.abs(X[:, s:e] - CF[:, s:e]).sum(1) > 1e-6
        out[f"viol_{name}"] = ch; br |= ch
    s, e = idx[age]
    low = (CF[:, s:e].argmax(1) < X[:, s:e].argmax(1)) if ds == "oulad" else \
        (CF[:, s:e] < X[:, s:e] - 1e-6).any(1)
    out["viol_age_decrease"] = low; br |= low
    exc = np.zeros(len(X), bool)
    for f, (st, en) in zip(dm.features, dm.features.feature_indices):
        cats = [str(c) for c in np.asarray(f.transformation.transformer.categories_).ravel()] \
            if f.is_categorical else []
        for lab in banned.get(f.name, []):
            exc |= CF[:, st + cats.index(str(lab))] > 0.5
    out["out_disallowed_any"] = exc; br |= exc
    out["viol_any"] = br
    return {k: float(v.mean()) for k, v in out.items()}


def setting(ds, X, blk, dm, d):
    summ = json.load(open(d / "summary.json"))["experiment_1"]  # experiment 1: the CT-trained model
    CF = R.snap(np.loadtxt(d / "experiment_1_counterfactuals.csv", delimiter=","), blk)
    # columns: factual prediction, target class, prediction of the raw and of the snapped CF
    P = np.loadtxt(d / "experiment_1_factual_pred_target.csv", delimiter=",")
    yx, yc = P[:, 0], P[:, 3]
    l1 = np.abs(CF - X).sum(1)
    rej = yx != R.FAV[ds]
    ok = rej & (yc != yx)
    n = nchg(X, CF, blk)
    return dict(acc=float(summ["accuracy"]), validity=float((yc != yx).mean()),
                recourse_validity=float(ok.sum() / rej.sum()), proximity=float(l1.mean()),
                any_cost=float(l1[rej].mean()), recourse_cost=float(l1[ok].mean()),
                n_changes=float(n[ok].mean()), seconds=float(summ["seconds"]),
                **rule_shares(ds, X, CF, dm))


def main():
    out = {}
    for ds, (dn, arms) in ARMS.items():
        dm = R.SPEC[ds]["load"](R.SPEC[ds]["fixed"])
        blk = R.blocks(dm)
        X = np.loadtxt(CT_DATA / ds / "X_test.csv", delimiter=",")
        assert np.allclose(X, np.asarray(dm["test"][0]), atol=1e-5), "CT export differs from CounterNet's test rows"
        out[ds] = {}
        for arm, path in arms.items():
            per = [setting(ds, X, blk, dm, CT_RUNS / path / "mlp" / dn / sub) for sub in SETTINGS]
            m = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "seconds"}
            m["time_ms"] = 1e3 * statistics.median(p["seconds"] for p in per)
            out[ds][arm] = m
        print(ds, {a: round(v["recourse_cost"], 3) for a, v in out[ds].items()}, flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
