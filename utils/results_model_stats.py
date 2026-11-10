#!/usr/bin/env python3
"""Model-based statistics for the Results chapter, cached to utils/stats/results_model_stats.json.

For each dataset, on the 8,141 / 7,500 / 8,149 test rows:
  * proximity split into its continuous and categorical parts, and the number of
    categorical features switched per person, over all test rows (``proximity_split``) and
    over the valid counterfactuals of the individuals each model rejects, the population of
    recourse cost (``recourse_split``), for CounterNet without constraints, the
    feasibility baseline, the best regularizer-A and regularizer-B cells, CT arm C (CT's original
    evaluation, averaged over its 10 search settings) and Model-approx with the feasibility
    set (C1, alpha = 1, seed 42, averaged over its 10 draws per person);
  * the share of people whose counterfactual switches each categorical feature, for the
    feasibility baseline and the best regularizer-A cell, over the same two populations
    (``switch_rates``, ``recourse_switch_rates``);
  * the Predictor's sensitivity to each feature, mean over test rows of the summed
    absolute (``sensitivity``) and squared (``sensitivity_sq``, the quantity regularizer B
    penalises) input gradient of the favourable-class probability over the feature's
    columns, for the feasibility baseline and the best regularizer-B cell.

Read-only with respect to all trained models. Run from jax-relax/ with
PYTHONPATH=. .venv/bin/python utils/results_model_stats.py [--t7]

With --t7, the CounterNet models are those trained with the CounterNet paper's Table 7
settings (table7_refs.py, table7_grid.py); the selected regularized cells are re-derived from
the Table 7 grid with the same selection rule, and the output is results_model_stats_t7.json.
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

import thesis_immutability_experiment as T
import ar_grid as A
import cred_grid as C
import oulad_grid as O
from relax.methods.counternet import CounterNet, CounterNetConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "utils/stats/results_model_stats.json"
OUT_T7 = ROOT / "utils/stats/results_model_stats_t7.json"
T7_GRID = ROOT / "trained_models/table7_grid"
T7_REFS = ROOT / "trained_models/table7_refs"
CT_RUNS = ROOT.parent / "ct_runs_cn"
CT_DATA = ROOT.parent / "ct_data"
FAV = {"adult": 1, "credit": 0, "oulad": 1}   # the favourable class: income above 50K, no default, pass
MA = ROOT / "trained_models"
MA_RUNS = {  # Model-approx C1 with the feasibility set (alpha = 1), seed 42: inputs, candidates
    "adult": (MA / "cf_feasibility_baseline/adult_inputs.npz",
              MA / "cf_feasibility_penalties/c1/loss_only_seed42/test_candidates.npz"),
    "credit": (MA / "cf_feasibility_multidata/credit/inputs.npz",
               MA / "cf_feasibility_multidata/credit/c1/alpha_1/seed_42/test_candidates.npz"),
    "oulad": (MA / "cf_feasibility_multidata/oulad/inputs.npz",
              MA / "cf_feasibility_multidata/oulad/c1/alpha_1/seed_42/test_candidates.npz"),
}

SPEC = {
    "adult": dict(load=T.load_dm, out=T.OUT_DIR, fixed=T.IMMUTABLE,
                  proj=A._proj_kwargs(),
                  unconstrained="actionability_l4_0.0",
                  baseline="reachability_projection_fixed3_age_nondec_no_unknown",
                  reg_a="actionability_recourse_finetune_everyk5_l4_0.5_20ep",
                  reg_b="actionability_recourse_gradpen_every_k_50_l4_5.0",
                  ct="cn_adult_ct_counternet_dec/ct_counternet/mlp/cn_adult"),
    "credit": dict(load=C.load_dm, out=C.OUT, fixed=C.FIXED, proj=C.proj_kwargs(),
                   unconstrained="cred_base", baseline="cred_reachability_baseline",
                   reg_a="cred_predprox_after_k_1760_l4_0.5",
                   reg_b="cred_gradpen_every_k_10_l4_10.0",
                   ct="cn_credit_ct_counternet_dec/ct_counternet/mlp/cn_credit"),
    "oulad": dict(load=O.load_dm, out=O.OUT, fixed=O.FIXED, proj=O.proj_kwargs(),
                  unconstrained="oulad_base", baseline="oulad_reachability_baseline",
                  reg_a="oulad_predprox_after_k_3820_l4_10.0",
                  reg_b="oulad_gradpen_after_k_1910_l4_0.5",
                  ct="cn_oulad_ct_counternet/ct_counternet/mlp/cn_oulad"),
}


def load_net(sp, name, projected, arch=None, path=None):
    """`path` overrides sp["out"] / name / "params.pkl"; `arch` adds Table 7 settings."""
    arch = arch or {}
    dm = sp["load"](sp["fixed"] if projected else [])
    if projected:
        cfg = T.ReachabilityProjectionConfig(**sp["proj"], **arch)
        net = CounterNet(cfg, cfnet_module=T.ReachabilityProjectionModule(cfg))
    else:
        net = CounterNet(CounterNetConfig(**arch))
    net.set_data_module(dm)
    net.module = net._init_model(net.config, net.module)
    net.module._data_module = dm
    net.params = pickle.load(open(path or sp["out"] / name / "params.pkl", "rb"))
    net._is_trained = True
    if projected:  # build the masks eagerly, outside any trace
        _ = (net.module.fixed_mask, net.module.nondecreasing_mask,
             net.module.disallowed_category_mask, net.module.ordinal_blocks)
    return net, dm


def blocks(dm):
    return [(f.name, bool(f.is_categorical), s, e)
            for f, (s, e) in zip(list(dm.features), dm.features.feature_indices)]


def split(X, CF, blk, rows=None):
    """Proximity split over all rows, or over `rows` (a boolean mask) only."""
    if rows is not None:
        X, CF = X[rows], CF[rows]
    d = np.abs(CF - X)
    cont = sum(d[:, s:e].sum(1) for _, cat, s, e in blk if not cat)
    cat = sum(d[:, s:e].sum(1) for _, c, s, e in blk if c)
    sw = {n: float((CF[:, s:e].argmax(1) != X[:, s:e].argmax(1)).mean())
          for n, c, s, e in blk if c}
    return dict(total=float(d.sum(1).mean()), continuous=float(cont.mean()),
                categorical=float(cat.mean()),
                switches_per_person=float(sum(sw.values()))), sw


def snap(CF, blk):
    CF = CF.copy()
    for _, c, s, e in blk:
        if c:
            a = CF[:, s:e].argmax(1)
            CF[:, s:e] = 0
            CF[np.arange(len(CF)), s + a] = 1
    return CF


def sensitivity(net, X, blk):
    def p1(x):
        return net.pred_fn(x)[:, 1].sum()
    g = np.asarray(jax.grad(p1)(jnp.asarray(X)))
    return ({n: float(np.abs(g[:, s:e]).sum(1).mean()) for n, _, s, e in blk},
            {n: float((g[:, s:e] ** 2).sum(1).mean()) for n, _, s, e in blk})


def t7_paths(ds):
    """Table 7 model files: the unconstrained reference, the feasibility baseline and the best
    cell of each regularizer under the main selection rule."""
    import pandas as pd
    b = pd.read_csv(T7_GRID / f"{ds}_baseline.csv").iloc[0]
    g = pd.read_csv(T7_GRID / f"{ds}_grid_results.csv")
    best = {}
    for loss, key in (("proximity", "reg_a"), ("gradient", "reg_b")):
        s = g[(g.loss == loss) & (g.validity >= b.validity - 0.002) & (g.acc >= b.acc - 0.005)]
        best[key] = s.sort_values("proximity").iloc[0].model
    un = {"adult": "adult/actionability_l4_0.0", "credit": "credit/credit_base",
          "oulad": "oulad/oulad_base"}[ds]
    return {"unconstrained": T7_REFS / un / "params.pkl",
            "baseline": T7_GRID / ds / "feasibility_baseline" / "params.pkl",
            "reg_a": T7_GRID / ds / best["reg_a"] / "params.pkl",
            "reg_b": T7_GRID / ds / best["reg_b"] / "params.pkl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t7", action="store_true")
    t7 = ap.parse_args().t7
    import table7_check as K
    out = {}
    for ds, sp in SPEC.items():
        res = dict(proximity_split={}, recourse_split={}, switch_rates={}, recourse_switch_rates={},
                   sensitivity={})
        if t7:
            paths, arch = t7_paths(ds), K.TABLE7[ds]
            nets = {k: load_net(sp, None, k != "unconstrained", arch, paths[k]) for k in paths}
            res["models"] = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
        else:
            nets = {"unconstrained": load_net(sp, sp["unconstrained"], False),
                    "baseline": load_net(sp, sp["baseline"], True),
                    "reg_a": load_net(sp, sp["reg_a"], True),
                    "reg_b": load_net(sp, sp["reg_b"], True)}
        _, dm = nets["baseline"]
        X = np.asarray(dm["test"][0])
        blk = blocks(dm)
        for key in ("unconstrained", "baseline", "reg_a", "reg_b"):
            net, dmk = nets[key]
            assert np.allclose(np.asarray(dmk["test"][0]), X)
            CF = np.asarray(net.module.generate_cf(net.params, jrand.PRNGKey(0), jnp.asarray(X)))
            yx = np.asarray(net.pred_fn(jnp.asarray(X))).argmax(-1)
            yc = np.asarray(net.pred_fn(jnp.asarray(CF))).argmax(-1)
            ok = (yx != FAV[ds]) & (yc != yx)            # valid counterfactuals of rejected people
            res["proximity_split"][key], sw = split(X, CF, blk)
            res["recourse_split"][key], rsw = split(X, CF, blk, ok)
            if key != "unconstrained":
                res["switch_rates"][key] = sw
                res["recourse_switch_rates"][key] = rsw
        # CT arm C, original evaluation (categories rounded after every step), 10 settings
        Xct = np.loadtxt(CT_DATA / ds / "X_test.csv", delimiter=",")
        assert np.allclose(Xct, X, atol=1e-5), "CT export differs from CounterNet's test rows"
        dirs = sorted(glob.glob(str(CT_RUNS / sp["ct"] / "eval_snapped*")))
        assert len(dirs) == 10, dirs
        parts, rparts = [], []
        for d in dirs:
            CFct = snap(np.loadtxt(f"{d}/experiment_1_counterfactuals.csv", delimiter=","), blk)
            # columns: factual prediction, target, prediction of the raw and of the snapped CF
            P = np.loadtxt(f"{d}/experiment_1_factual_pred_target.csv", delimiter=",")
            ok = (P[:, 0] != FAV[ds]) & (P[:, 3] != P[:, 0])
            parts.append(split(X, CFct, blk)[0])
            rparts.append(split(X, CFct, blk, ok)[0])
        res["proximity_split"]["ct_arm_c"] = {k: float(np.mean([p[k] for p in parts]))
                                             for k in parts[0]}
        res["recourse_split"]["ct_arm_c"] = {k: float(np.mean([p[k] for p in rparts]))
                                            for k in rparts[0]}
        # Model-approx: every draw counts, as in its results-table rows
        inputs, cands = (np.load(f) for f in MA_RUNS[ds])
        assert np.array_equal(cands["test_indices"], inputs["test_indices"])
        assert np.allclose(inputs["x"][inputs["test_indices"]], X), "Model-approx rows differ"
        CFma = cands["cf_decoded"]                       # draws x rows x columns
        Xma, CFma = np.tile(X, (len(CFma), 1)), CFma.reshape(-1, X.shape[1])
        fact = np.tile(cands["factual_prediction"], len(cands["cf_decoded"]))
        ok = (fact != FAV[ds]) & (cands["prediction_decoded"].reshape(-1) != fact)
        res["proximity_split"]["model_approx"] = split(Xma, CFma, blk)[0]
        res["recourse_split"]["model_approx"] = split(Xma, CFma, blk, ok)[0]
        res["sensitivity_sq"] = {}
        for key in ("baseline", "reg_b"):
            res["sensitivity"][key], res["sensitivity_sq"][key] = sensitivity(nets[key][0], X, blk)
        res["immutable"] = list(sp["fixed"])
        out[ds] = res
        print(ds, json.dumps(res["recourse_split"], indent=1))
    (OUT_T7 if t7 else OUT).write_text(json.dumps(out, indent=1))
    print("wrote", OUT_T7 if t7 else OUT)


if __name__ == "__main__":
    main()
