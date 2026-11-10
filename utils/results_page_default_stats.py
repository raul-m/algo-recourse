"""Results-page statistics for the default CounterNet configuration (the section above the
"CounterNet Table 7 grid" section of overall_results.html): every column of the per-dataset
tables for the feasibility baseline and the best and the median of each regularizer's 90 cells,
under the same selection rule and metric definitions as results_page_t7_stats.py. Read-only.

    PYTHONPATH=. .venv/bin/python utils/results_page_default_stats.py      (from the jax-relax root)

Writes utils/stats/results_page_default_stats.json. The CF generation time is measured again
(time_ms_remeasured) only as a check: the page keeps the times recorded when the models were
evaluated, since timings vary between runs and machines.
"""
import json, pickle, sys, time
from pathlib import Path

import jax.numpy as jnp, jax.random as jrand
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_model_stats as R

ROOT = R.ROOT
OUT = ROOT / "utils/stats/results_page_default_stats.json"
E = ROOT / "trained_models/immutability_experiment"
BASELINE = {  # the feasibility baseline's row of each dataset's reference table
    "adult": (E / "results.csv", "variant",
              "7: reachability projection (fixed 3, age non-decreasing, no mutable unknown)"),
    "credit": (E / "cred/cred_reference.csv", "model", "cred_reachability_baseline"),
    "oulad": (E / "oulad/oulad_reference.csv", "model", "oulad_reachability_baseline"),
}
GRID = {"adult": E / "ar_grid_results.csv", "credit": E / "cred/cred_grid_results.csv",
        "oulad": E / "oulad/oulad_grid_results.csv"}
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


def holds(df, b):
    return df[(df.validity >= b.validity - 0.002) & (df.acc >= b.acc - 0.005)]


def main():
    out = {}
    for ds, sp in R.SPEC.items():
        net, dm = R.load_net(sp, sp["baseline"], True)
        X = np.asarray(dm["test"][0]); Xj = jnp.asarray(X); blk = R.blocks(dm)
        path, col, value = BASELINE[ds]
        r = pd.read_csv(path)
        b = r[r[col] == value].iloc[0]
        base = dict(b.drop(labels=[col]).to_dict(), **stats(net, X, Xj, blk, R.FAV[ds]),
                    time_ms_remeasured=timing(net, Xj))
        g = pd.read_csv(GRID[ds])
        assert len(g) == 180, len(g)
        extra = []
        for name in g.model:
            net.params = pickle.load(open(sp["out"] / name / "params.pkl", "rb"))
            extra.append(stats(net, X, Xj, blk, R.FAV[ds]))
        g = pd.concat([g, pd.DataFrame(extra)], axis=1)
        res = {"baseline": base}
        for loss, reg in (("proximity", "A"), ("gradient", "B")):
            s = g[g.loss == loss]
            h = holds(s, b)
            best = h.sort_values("proximity").iloc[0]
            net.params = pickle.load(open(sp["out"] / best.model / "params.pkl", "rb"))
            res[f"{reg}_best"] = dict(best.to_dict(), n_hold=int(len(h)), time_ms_remeasured=timing(net, Xj))
            num = s.select_dtypes("number")
            res[f"{reg}_median"] = {c: float(num[c].median()) for c in num.columns}
        out[ds] = res
        print(ds, {k: round(v.get("proximity", float("nan")), 3) for k, v in res.items()}, flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
