"""Actionability-recourse grid on the Open University Learning Analytics dataset (OULAD).

Constraint design (mirrors the Adult and Cred studies' structure):
  fixed (copied)        : gender, region, disability          -> 17 one-hot columns
  non-decreasing        : age_band  (ordinal one-hot: 0-35 < 35-55 < 55<=)
  banned as CE target   : imd_band == '0'  (OULAD's missing-IMD placeholder, 3.3% of rows)

`age_band` is a 3-level one-hot rather than a scalar, so it uses the ordinal
monotone projection (`nondecreasing_ordinal_features`) rather than the continuous
`jnp.maximum` lower bound used for Adult's and Cred's numeric age columns.

Label direction: relax ships `final_result` binarised as 1 = Pass/Distinction,
0 = Fail/Withdrawn -- i.e. Y=1 is the *favourable* outcome, matching Adult and Cred.
(The CounterNet paper's prose describes OULAD as predicting dropout with Y=1; the
shipped data is the other way round.)

Reference rows: base CounterNet; feasibility (gender only -- the only immutable
feature the CounterNet paper names, §4.2); projection (gender+region+disability);
reachability projection = the lambda4 = 0 baseline every E-cell is scored against.
"""
import argparse, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, optax
import jax, jax.numpy as jnp, jax.random as jrand
import relax
import thesis_immutability_experiment as T
from relax.methods.counternet import CounterNet
from relax.evaluate import compute_validity, compute_proximity, compute_sparsity

DATA = "oulad"
FIXED = ["gender", "region", "disability"]
AGE = "age_band"                      # ordinal one-hot, ascending
BANNED = {"imd_band": ["0"]}
OUT = T.OUT_DIR / "oulad"
OUT.mkdir(parents=True, exist_ok=True)
GRID_CSV = OUT / "oulad_grid_results.csv"
REF_CSV = OUT / "oulad_reference.csv"
LAMBDAS = [0.1, 0.5, 1.0, 5.0, 10.0]
K_STEPS = [5, 10, 20, 50]
K_EPOCHS_JOINT = [10, 20, 50, 80]
K_EPOCHS_FT = [2, 5, 10, 15]


def load_dm(immutable=()):
    dm = relax.load_data(DATA)
    for f in dm.features:
        f._is_immutable = f.name in immutable
    return dm


def proj_kwargs():
    return dict(fixed_features=FIXED, nondecreasing_ordinal_features=[AGE],
                disallowed_categories=BANNED)


def evaluate(cfnet, dm) -> dict:
    """Same metric set as the Adult and Cred studies, with rank-based age metrics."""
    xs, ys = dm["test"]; d = xs.shape[1]
    cfs = cfnet.module.generate_cf(cfnet.params, jrand.PRNGKey(0), xs)
    preds = cfnet.pred_fn(xs); cf_preds = cfnet.pred_fn(cfs)
    mask = T.feature_col_mask(dm, FIXED)
    n_mut = d - int(mask.sum())
    perm = jrand.permutation(jrand.PRNGKey(1), xs.shape[0])
    x_swap = jnp.where(mask.astype(bool), xs[perm], xs)
    p1 = preds[:, 1:2]; y_true = jnp.asarray(ys).reshape(-1, 1).astype(p1.dtype)
    row = {
        "n_test": int(xs.shape[0]),
        "acc": float((jnp.argmax(preds, -1) == jnp.asarray(ys).reshape(-1)).mean()),
        "pred_gap": float(jnp.abs(preds[:, 1] - cfnet.pred_fn(x_swap)[:, 1]).mean()),
        "validity": float(compute_validity(xs, cfs, cfnet.pred_fn)),
        "L1": float(jnp.mean(optax.l2_loss(p1, y_true))),
        "L2": float(jnp.mean(optax.l2_loss(cf_preds[:, 1:2], 1.0 - jnp.round(p1)))),
        "L3": float(jnp.mean(optax.l2_loss(cfs, xs))),
        "proximity": float(compute_proximity(xs, cfs)),
    }
    row["L4"] = T._predictor_l4(cfnet, dm, xs, row["L3"])
    row["prox_norm"] = row["proximity"] / d
    row["sparsity"] = float(compute_sparsity(xs, cfs, dm.features.feature_indices))
    pm = float(jnp.abs((xs - cfs) * (1 - mask)).sum(axis=1).mean())
    row["prox_mutable_norm"] = pm / n_mut
    idx = {f.name: r for f, r in zip(dm.features, dm.features.feature_indices)}
    any_ch = jnp.zeros(xs.shape[0], dtype=bool)
    for name in FIXED:
        s, e = idx[name]
        ch = jnp.abs(xs[:, s:e] - cfs[:, s:e]).sum(axis=1) > 1e-6
        row[f"viol_{name}"] = float(ch.mean()); any_ch |= ch
    row["viol_any"] = float(any_ch.mean())
    # age_band is one-hot: compare selected category ranks, not raw column values
    s, e = idx[AGE]
    rank_x = jnp.argmax(xs[:, s:e], axis=1); rank_c = jnp.argmax(cfs[:, s:e], axis=1)
    row["chg_age"] = float((rank_c != rank_x).mean())
    row["viol_age_decrease"] = float((rank_c < rank_x).mean())
    dis = jnp.zeros(xs.shape[0], dtype=bool)
    for feat, (st, en) in zip(dm.features, dm.features.feature_indices):
        for lab in BANNED.get(feat.name, []):
            cats = [str(c) for c in
                    np.asarray(feat.transformation.transformer.categories_).ravel()]
            dis |= cfs[:, st + cats.index(str(lab))] > 0.5
    row["out_disallowed_any"] = float(dis.mean())
    return row


def joint_model(loss_kind, schedule, k, lam, spe):
    # after_k is specified in EPOCHS; the scheduler gate counts training STEPS.
    k = k * spe if schedule == "after_k" else k
    if loss_kind == "proximity":
        if schedule == "always":
            cfg = T.PredictorProximityConfig(**proj_kwargs(), lambda_4=lam)
            return cfg, T.PredictorProximityReachabilityModule(cfg)
        cfg = T.ScheduledPredictorProximityConfig(**proj_kwargs(), lambda_4=lam,
                                                  l4_schedule=schedule, l4_k=k)
        return cfg, T.ScheduledPredictorProximityModule(cfg)
    if schedule == "always":
        cfg = T.ImmutableGradientConfig(**proj_kwargs(), invariant_features=FIXED,
                                        lambda_4=lam)
        return cfg, T.ImmutableGradientPredictorModule(cfg)
    cfg = T.ScheduledImmutableGradientConfig(**proj_kwargs(), invariant_features=FIXED,
                                             lambda_4=lam, l4_schedule=schedule, l4_k=k)
    return cfg, T.ScheduledImmutableGradientModule(cfg)


def name_joint(loss_kind, schedule, k, lam, spe):
    stem = "predprox" if loss_kind == "proximity" else "gradpen"
    if schedule == "always":   return f"oulad_{stem}_l4_{lam}"
    if schedule == "every_k":  return f"oulad_{stem}_every_k_{k}_l4_{lam}"
    return f"oulad_{stem}_after_k_{k*spe}_l4_{lam}"


def name_ft(loss_kind, schedule, k, lam):
    stem = "finetune" if loss_kind == "proximity" else "gradfinetune"
    if schedule == "always": return f"oulad_{stem}_l4_{lam}_{T.FINETUNE_EPOCHS}ep"
    tag = f"everyk{k}" if schedule == "every_k" else f"afterk{k}"
    return f"oulad_{stem}_{tag}_l4_{lam}_{T.FINETUNE_EPOCHS}ep"


def hydrate(name, cfg, module, dm):
    net = CounterNet(cfg, cfnet_module=module)
    net.set_data_module(dm)
    net.module = net._init_model(net.config, net.module)
    net.module._data_module = dm
    net.params = pickle.load(open(OUT / name / "params.pkl", "rb"))
    net._is_trained = True
    return net


def train_and_cache(name, cfg, module, dm):
    net = CounterNet(cfg, cfnet_module=module)
    t = time.time()
    net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
    print(f"    trained {name} in {time.time()-t:.0f}s", flush=True)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    pickle.dump(net.params, open(OUT / name / "params.pkl", "wb"))
    return net


BASELINE_NAME = "oulad_reachability_baseline"


def baseline_net(dm):
    cfg = T.ReachabilityProjectionConfig(**proj_kwargs())
    module = T.ReachabilityProjectionModule(cfg)
    if (OUT / BASELINE_NAME / "params.pkl").exists():
        return hydrate(BASELINE_NAME, cfg, module, dm)
    return train_and_cache(BASELINE_NAME, cfg, module, dm)


def references(rows):
    """base / feasibility(gender) / projection(gender+region+disability) / reachability."""
    specs = [
        ("oulad_base", [], "base CounterNet (no enforcement)"),
        ("oulad_feasibility_gender", ["gender"],
         "feasibility CounterNet (gender immutable &mdash; the paper's &sect;4.2 example)"),
        ("oulad_projection", FIXED, "projection (gender + region + disability)"),
    ]
    for name, immut, label in specs:
        dm = load_dm(immut)
        cfg = T.CounterNetConfig()
        if (OUT / name / "params.pkl").exists():
            net = CounterNet(cfg); net.set_data_module(dm)
            net.module = net._init_model(net.config, net.module)
            net.module._data_module = dm
            net.params = pickle.load(open(OUT / name / "params.pkl", "rb"))
            net._is_trained = True
        else:
            net = CounterNet(cfg); t = time.time()
            net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
            print(f"    trained {name} in {time.time()-t:.0f}s", flush=True)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            pickle.dump(net.params, open(OUT / name / "params.pkl", "wb"))
        r = evaluate(net, dm); r.update(model=name, label=label); rows.append(r)
    dm = load_dm(FIXED)
    net = baseline_net(dm)
    r = evaluate(net, dm)
    r.update(model=BASELINE_NAME,
             label="reachability projection &mdash; the &lambda;4 = 0 baseline")
    rows.append(r)


def run_cell(loss_kind, phase, schedule, k, lam, spe, rows):
    dm = load_dm(FIXED)
    if phase == "joint":
        name = name_joint(loss_kind, schedule, k, lam, spe)
        cfg, module = joint_model(loss_kind, schedule, k, lam, spe)
        net = (hydrate(name, cfg, module, dm) if (OUT / name / "params.pkl").exists()
               else train_and_cache(name, cfg, module, dm))
    else:
        name = name_ft(loss_kind, schedule, k, lam)
        cfg = T.ReachabilityProjectionConfig(**proj_kwargs())
        module = T.ReachabilityProjectionModule(cfg)
        if (OUT / name / "params.pkl").exists():
            net = hydrate(name, cfg, module, dm)
        else:
            net = baseline_net(dm)
            t = time.time()
            net = T.finetune_predictor_on_l4(
                net, dm, lambda_4=lam, epochs=T.FINETUNE_EPOCHS,
                every_k=(k if schedule == "every_k" else 1),
                after_k=(k * spe if schedule == "after_k" else 0),
                l4_kind=loss_kind, imm_features=FIXED)
            print(f"    fine-tuned {name} in {time.time()-t:.0f}s", flush=True)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            pickle.dump(net.params, open(OUT / name / "params.pkl", "wb"))
    r = evaluate(net, dm)
    r.update(loss=loss_kind, phase=phase, schedule=schedule,
             k=(k if schedule != "always" else 0), lambda_4=lam, model=name)
    rows.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-only", action="store_true",
                    help="train/evaluate the four reference rows and stop")
    ap.add_argument("--smoke", action="store_true",
                    help="one cell per (loss, phase) to check the plumbing")
    args = ap.parse_args()

    spe = int(np.ceil(len(load_dm()["train"][0]) / T.BATCH_SIZE))
    print(f"steps per epoch: {spe}", flush=True)
    ref = []; references(ref)
    pd.DataFrame(ref).to_csv(REF_CSV, index=False)
    print(pd.DataFrame(ref)[["label", "acc", "validity", "proximity", "prox_norm",
                             "viol_any", "viol_age_decrease",
                             "out_disallowed_any"]].to_string(index=False), flush=True)
    if args.refs_only:
        return

    rows = []
    if args.smoke:
        for loss_kind in ("proximity", "gradient"):
            for phase in ("joint", "two-phase"):
                run_cell(loss_kind, phase, "always", 0, 1.0, spe, rows)
        print(pd.DataFrame(rows)[["model", "acc", "validity", "proximity",
                                  "viol_any"]].to_string(index=False))
        return

    for loss_kind in ("proximity", "gradient"):
        for phase in ("joint", "two-phase"):
            print(f"== {loss_kind} / {phase} ==", flush=True)
            for lam in LAMBDAS:
                run_cell(loss_kind, phase, "always", 0, lam, spe, rows)
            for k in K_STEPS:
                for lam in LAMBDAS:
                    run_cell(loss_kind, phase, "every_k", k, lam, spe, rows)
            for k in (K_EPOCHS_JOINT if phase == "joint" else K_EPOCHS_FT):
                for lam in LAMBDAS:
                    run_cell(loss_kind, phase, "after_k", k, lam, spe, rows)
            pd.DataFrame(rows).to_csv(GRID_CSV, index=False)
    df = pd.DataFrame(rows); df.to_csv(GRID_CSV, index=False)
    print(f"\n{len(df)} cells -> {GRID_CSV}")


if __name__ == "__main__":
    main()
