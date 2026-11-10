"""Results-page statistics for the Table 7 grid: every column of the per-dataset tables for the
Table 7 feasibility baseline, the best and the median of each regularizer's 90 cells, and CounterNet
without constraints. Read-only.

    .venv/bin/python -I utils/results_page_t7_stats.py utils/stats/results_page_t7_stats.json      (run from the jax-relax root)

With --unconstrained, only the entry of CounterNet without constraints is computed and added to
the existing out.json, so that the other entries, and their timings, stay as they are.
"""
import json, pickle, sys, time
import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jrand

sys.path.insert(0, ".")
sys.path.insert(0, "utils")
import table7_check as K
import results_model_stats as R

G = "trained_models/table7_grid"
TH = 1e-2  # a continuous feature counts as changed above 1% of its range


def nchg(X, CF, blk):
    n = np.zeros(len(X))
    for _, cat, s, e in blk:
        n += (CF[:, s:e].argmax(1) != X[:, s:e].argmax(1)) if cat else (np.abs(CF[:, s] - X[:, s]) > TH)
    return n


def stats(net, X, Xj, blk, fav):
    CF = np.asarray(net.module.generate_cf(net.params, jrand.PRNGKey(0), Xj))
    yx = np.asarray(net.pred_fn(Xj)).argmax(-1)
    yc = np.asarray(net.pred_fn(jnp.asarray(CF))).argmax(-1)
    l1 = np.abs(CF - X).sum(1)
    rej = yx != fav
    ok = rej & (yc != yx)
    n = nchg(X, CF, blk)
    nan = float("nan")
    return dict(recourse_validity=float(ok.sum() / rej.sum()) if rej.any() else nan,
                any_cost=float(l1[rej].mean()) if rej.any() else nan,
                recourse_cost=float(l1[ok].mean()) if ok.any() else nan,
                n_changes=float(n[ok].mean()) if ok.any() else nan)


def timing(net, Xj, runs=30):
    f = lambda: net.module.generate_cf(net.params, jrand.PRNGKey(0), Xj).block_until_ready()
    f()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def any_rule(ds, X, CF, dm):
    """Share of counterfactuals breaking at least one rule of the feasibility set: an immutable
    feature changed, age lowered (OULAD: a lower age band), or an excluded category chosen, each
    as the dataset's evaluator counts it."""
    fixed, age, banned = {"adult": (R.T.IMMUTABLE, R.T.AGE, R.T.DISALLOWED_MUTABLE_CATEGORIES),
                          "credit": (R.C.FIXED, R.C.AGE, R.C.BANNED),
                          "oulad": (R.O.FIXED, R.O.AGE, R.O.BANNED)}[ds]
    idx = {f.name: r for f, r in zip(dm.features, dm.features.feature_indices)}
    br = np.zeros(len(X), bool)
    for name in fixed:
        s, e = idx[name]
        br |= np.abs(X[:, s:e] - CF[:, s:e]).sum(1) > 1e-6
    s, e = idx[age]
    br |= (CF[:, s:e].argmax(1) < X[:, s:e].argmax(1)) if ds == "oulad" else \
        (CF[:, s:e] < X[:, s:e] - 1e-6).any(1)
    for f, (st, en) in zip(dm.features, dm.features.feature_indices):
        cats = [str(c) for c in np.asarray(f.transformation.transformer.categories_).ravel()] \
            if f.is_categorical else []
        for lab in banned.get(f.name, []):
            br |= CF[:, st + cats.index(str(lab))] > 0.5
    return float(br.mean())


def unconstrained(ds, sp):
    """CounterNet without constraints, Table 7 settings: the evaluator's row, with the share
    breaking any rule as its viol_any, plus the recourse statistics and the timing."""
    net, dm = R.load_net(R.SPEC[ds], None, False, K.TABLE7[ds], R.t7_paths(ds)["unconstrained"])
    X = np.asarray(dm["test"][0]); Xj = jnp.asarray(X)
    row = sp["evaluate"](net, dm)
    CF = np.asarray(net.module.generate_cf(net.params, jrand.PRNGKey(0), Xj))
    row["viol_any_immutable"], row["viol_any"] = row["viol_any"], any_rule(ds, X, CF, dm)
    return dict(row, **stats(net, X, Xj, R.blocks(dm), K.FAV[ds]), time_ms=timing(net, Xj))


def holds(df, b):
    return df[(df.validity >= b.validity - 0.002) & (df.acc >= b.acc - 0.005)]


if "--unconstrained" in sys.argv:
    out = json.load(open(sys.argv[1]))
    for ds, sp in K.SPEC.items():
        out[ds]["unconstrained"] = unconstrained(ds, sp)
        print(ds, {k: round(out[ds]["unconstrained"][k], 4) for k in
                   ("acc", "validity", "proximity", "recourse_validity", "recourse_cost", "viol_any")})
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    sys.exit()

out = {}
for ds, sp in K.SPEC.items():
    arch = K.TABLE7[ds]
    dm = sp["load"]()
    X = np.asarray(dm["test"][0]); Xj = jnp.asarray(X); blk = R.blocks(dm)
    cfg, mod = K.baseline_cfg(sp, arch)
    net = K.hydrate(f"{G}/{ds}/feasibility_baseline/params.pkl", cfg, mod, dm)
    b = pd.read_csv(f"{G}/{ds}_baseline.csv").iloc[0]
    base = dict(b.to_dict(), **stats(net, X, Xj, blk, K.FAV[ds]), time_ms=timing(net, Xj))
    g = pd.read_csv(f"{G}/{ds}_grid_results.csv")
    extra = []
    for name in g.model:
        net.params = pickle.load(open(f"{G}/{ds}/{name}/params.pkl", "rb"))
        extra.append(stats(net, X, Xj, blk, K.FAV[ds]))
    g = pd.concat([g.drop(columns=[c for c in ("recourse_validity", "recourse_cost") if c in g]),
                   pd.DataFrame(extra)], axis=1)
    res = {"baseline": base, "unconstrained": unconstrained(ds, sp)}
    for loss, reg in (("proximity", "A"), ("gradient", "B")):
        s = g[g.loss == loss]
        h = holds(s, b)
        best = h.sort_values("proximity").iloc[0]
        net.params = pickle.load(open(f"{G}/{ds}/{best.model}/params.pkl", "rb"))
        res[f"{reg}_best"] = dict(best.to_dict(), time_ms=timing(net, Xj), n_hold=int(len(h)))
        num = s.select_dtypes("number")
        res[f"{reg}_median"] = {c: float(num[c].median()) for c in num.columns}
        res[f"{reg}_median"]["n_no_rejected"] = int(s.recourse_validity.isna().sum())
    out[ds] = res
    print(ds, {k: round(v.get("proximity", float("nan")), 3) for k, v in res.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
