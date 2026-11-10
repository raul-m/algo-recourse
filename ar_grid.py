"""Actionability-recourse grid: loss kind x schedule x phase mode x lambda4 x k.

Reuses every model already cached by thesis_immutability_experiment.py (naming is
kept compatible), so re-running only trains the new cells. Writes its own CSV.
"""
import argparse, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, jax.numpy as jnp
import thesis_immutability_experiment as T
from relax.methods.counternet import CounterNet

OUT = T.OUT_DIR
GRID_CSV = OUT / "ar_grid_results.csv"
LAMBDAS = [0.1, 0.5, 1.0, 5.0, 10.0]
K_STEPS = [5, 10, 20, 50]
K_EPOCHS_JOINT = [10, 20, 50, 80]
K_EPOCHS_FT = [2, 5, 10, 15]


def _proj_kwargs():
    return dict(fixed_features=T.IMMUTABLE, nondecreasing_features=[T.AGE],
                disallowed_categories=T.DISALLOWED_MUTABLE_CATEGORIES)


def joint_model(loss_kind, schedule, k, lam, spe):
    # after_k is specified in EPOCHS; the scheduler gate counts training STEPS.
    k = k * spe if schedule == "after_k" else k
    """Config+module for a single-phase (joint) cell."""
    if loss_kind == "proximity":
        if schedule == "always":
            cfg = T.PredictorProximityConfig(**_proj_kwargs(), lambda_4=lam)
            return cfg, T.PredictorProximityReachabilityModule(cfg)
        cfg = T.ScheduledPredictorProximityConfig(
            **_proj_kwargs(), lambda_4=lam, l4_schedule=schedule, l4_k=k)
        return cfg, T.ScheduledPredictorProximityModule(cfg)
    if schedule == "always":
        cfg = T.ImmutableGradientConfig(**_proj_kwargs(),
                                        invariant_features=T.IMMUTABLE, lambda_4=lam)
        return cfg, T.ImmutableGradientPredictorModule(cfg)
    cfg = T.ScheduledImmutableGradientConfig(
        **_proj_kwargs(), invariant_features=T.IMMUTABLE, lambda_4=lam,
        l4_schedule=schedule, l4_k=k)
    return cfg, T.ScheduledImmutableGradientModule(cfg)


def name_joint(loss_kind, schedule, k, lam, spe):
    stem = "predprox" if loss_kind == "proximity" else "gradpen"
    if schedule == "always":
        return f"actionability_recourse_{stem}_l4_{lam}"
    if schedule == "every_k":
        return f"actionability_recourse_{stem}_every_k_{k}_l4_{lam}"
    return f"actionability_recourse_{stem}_after_k_{k * spe}_l4_{lam}"


def name_ft(loss_kind, schedule, k, lam):
    stem = "finetune" if loss_kind == "proximity" else "gradfinetune"
    if schedule == "always":
        return f"actionability_recourse_{stem}_l4_{lam}_{T.FINETUNE_EPOCHS}ep"
    tag = f"everyk{k}" if schedule == "every_k" else f"afterk{k}"
    return f"actionability_recourse_{stem}_{tag}_l4_{lam}_{T.FINETUNE_EPOCHS}ep"


def load_cached(name, cfg, module, dm):
    net = CounterNet(cfg, cfnet_module=module)
    net.set_data_module(dm)
    net.module = net._init_model(net.config, net.module)
    net.module._data_module = dm
    net.params = pickle.load(open(OUT / name / "params.pkl", "rb"))
    net._is_trained = True
    return net


def run_cell(loss_kind, phase, schedule, k, lam, spe, rows):
    dm = T.load_dm(T.IMMUTABLE)
    if phase == "joint":
        name = name_joint(loss_kind, schedule, k, lam, spe)
        cfg, module = joint_model(loss_kind, schedule, k, lam, spe)
        if (OUT / name / "params.pkl").exists():
            net = load_cached(name, cfg, module, dm)
        else:
            net = CounterNet(cfg, cfnet_module=module)
            t = time.time()
            net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
            print(f"    trained {name} in {time.time()-t:.0f}s", flush=True)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            pickle.dump(net.params, open(OUT / name / "params.pkl", "wb"))
    else:
        name = name_ft(loss_kind, schedule, k, lam)
        cfg = T.ReachabilityProjectionConfig(**_proj_kwargs())
        module = T.ReachabilityProjectionModule(cfg)
        if (OUT / name / "params.pkl").exists():
            net = load_cached(name, cfg, module, dm)
        else:
            net = T.get_or_train(
                "reachability_projection_fixed3_age_nondec_no_unknown", dm,
                CounterNet(cfg, cfnet_module=module))
            t = time.time()
            net = T.finetune_predictor_on_l4(
                net, dm, lambda_4=lam, epochs=T.FINETUNE_EPOCHS,
                every_k=(k if schedule == "every_k" else 1),
                after_k=(k * spe if schedule == "after_k" else 0),
                l4_kind=loss_kind)
            print(f"    fine-tuned {name} in {time.time()-t:.0f}s", flush=True)
            (OUT / name).mkdir(parents=True, exist_ok=True)
            pickle.dump(net.params, open(OUT / name / "params.pkl", "wb"))
    r = T.evaluate_variant(net, dm)
    r.update(loss=loss_kind, phase=phase, schedule=schedule, k=(k if schedule != "always" else 0),
             lambda_4=lam, model=name)
    rows.append(r)


def main(quick=False):
    dm0 = T.load_dm([])
    spe = int(np.ceil(len(dm0["train"][0]) / T.BATCH_SIZE))
    lams = [0.1, 1.0, 10.0] if quick else LAMBDAS
    ks_s = [10, 20] if quick else K_STEPS
    ks_e = [20, 50] if quick else K_EPOCHS_JOINT
    ks_f = [5, 10] if quick else K_EPOCHS_FT
    rows = []
    for loss_kind in ("proximity", "gradient"):
        for phase in ("joint", "two-phase"):
            print(f"== {loss_kind} / {phase} ==", flush=True)
            for lam in lams:
                run_cell(loss_kind, phase, "always", 0, lam, spe, rows)
            for k in ks_s:
                for lam in lams:
                    run_cell(loss_kind, phase, "every_k", k, lam, spe, rows)
            for k in (ks_e if phase == "joint" else ks_f):
                for lam in lams:
                    run_cell(loss_kind, phase, "after_k", k, lam, spe, rows)
    df = pd.DataFrame(rows)
    df.to_csv(GRID_CSV, index=False)
    print(f"\n{len(df)} cells -> {GRID_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
