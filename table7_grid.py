"""The full regularizer grid (180 cells per dataset) with the CounterNet paper's Table 7 settings.

Same cells, schedules, weights, training and evaluation as ar_grid.py, cred_grid.py and
oulad_grid.py; only the architecture and learning rate follow Guo et al.'s Table 7 (see
table7_check.TABLE7). Each cell is scored against the Table 7 feasibility baseline trained
by table7_check.py, which is copied in so that the check and the grid share one baseline.

    python table7_grid.py credit      # one dataset per run: adult, credit or oulad

Writes only to trained_models/table7_grid/. A fingerprint of the existing result folders
is taken before and after the run to confirm they are untouched.
"""
import pickle, shutil, sys, time
from pathlib import Path

import numpy as np, pandas as pd

import thesis_immutability_experiment as T
import ar_grid as A
import cred_grid as C
import oulad_grid as O
import table7_check as K

OUT = Path("trained_models/table7_grid")
PROTECTED = K.PROTECTED + [K.OUT]
for _p in PROTECTED:
    assert OUT.resolve() != _p.resolve() and _p.resolve() not in OUT.resolve().parents

NAMES = {"adult": A, "credit": C, "oulad": O}  # cell naming of the default grids
LAMBDAS = [0.1, 0.5, 1.0, 5.0, 10.0]
K_STEPS = [5, 10, 20, 50]
K_EPOCHS_JOINT = [10, 20, 50, 80]
K_EPOCHS_FT = [2, 5, 10, 15]


def joint_cfg(sp, loss, schedule, k, lam, spe, arch):
    """As joint_model in the default grids, with the Table 7 architecture added."""
    kk = k * spe if schedule == "after_k" else k
    if loss == "proximity":
        if schedule == "always":
            cfg = T.PredictorProximityConfig(**sp["proj"], lambda_4=lam, **arch)
            return cfg, T.PredictorProximityReachabilityModule(cfg)
        cfg = T.ScheduledPredictorProximityConfig(
            **sp["proj"], lambda_4=lam, l4_schedule=schedule, l4_k=kk, **arch)
        return cfg, T.ScheduledPredictorProximityModule(cfg)
    if schedule == "always":
        cfg = T.ImmutableGradientConfig(
            **sp["proj"], invariant_features=sp["fixed"], lambda_4=lam, **arch)
        return cfg, T.ImmutableGradientPredictorModule(cfg)
    cfg = T.ScheduledImmutableGradientConfig(
        **sp["proj"], invariant_features=sp["fixed"], lambda_4=lam,
        l4_schedule=schedule, l4_k=kk, **arch)
    return cfg, T.ScheduledImmutableGradientModule(cfg)


def cells():
    for loss in ("proximity", "gradient"):
        for phase in ("joint", "two-phase"):
            for lam in LAMBDAS:
                yield loss, phase, "always", 0, lam
            for k in K_STEPS:
                for lam in LAMBDAS:
                    yield loss, phase, "every_k", k, lam
            for k in (K_EPOCHS_JOINT if phase == "joint" else K_EPOCHS_FT):
                for lam in LAMBDAS:
                    yield loss, phase, "after_k", k, lam


def full_row(net, dm, sp, ds):
    r = sp["evaluate"](net, dm)
    r.update(K.recourse(net, dm, K.FAV[ds]))
    return r


def main(ds):
    sp, arch, names = K.SPEC[ds], K.TABLE7[ds], NAMES[ds]
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    before = K.fingerprint(PROTECTED)

    dm = sp["load"]()
    spe = int(np.ceil(len(dm["train"][0]) / T.BATCH_SIZE))
    print(f"{ds}: steps per epoch {spe}, Table 7 settings {arch}", flush=True)

    # the Table 7 feasibility baseline from table7_check.py, shared with the check
    base_path = out / "feasibility_baseline" / "params.pkl"
    if not base_path.exists():
        src = K.OUT / ds / "feasibility_baseline" / "params.pkl"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, base_path)
    cfg, mod = K.baseline_cfg(sp, arch)
    base = K.hydrate(base_path, cfg, mod, dm)
    b = full_row(base, dm, sp, ds)
    b.update(model="feasibility_baseline")
    pd.DataFrame([b]).to_csv(OUT / f"{ds}_baseline.csv", index=False)

    rows, todo = [], list(cells())
    for i, (loss, phase, schedule, k, lam) in enumerate(todo, 1):
        if phase == "joint":
            name = names.name_joint(loss, schedule, k, lam, spe)
            cfg, mod = joint_cfg(sp, loss, schedule, k, lam, spe, arch)
            path = out / name / "params.pkl"
            if path.exists():
                net = K.hydrate(path, cfg, mod, dm)
            else:
                from relax.methods.counternet import CounterNet
                net = CounterNet(cfg, cfnet_module=mod)
                t = time.time()
                net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(net.params, f)
                print(f"[{i}/{len(todo)}] trained {name} in {time.time() - t:.0f}s",
                      flush=True)
        else:
            name = names.name_ft(loss, schedule, k, lam)
            cfg, mod = K.baseline_cfg(sp, arch)
            path = out / name / "params.pkl"
            if path.exists():
                net = K.hydrate(path, cfg, mod, dm)
            else:
                # fine-tune a fresh copy of the Table 7 baseline (the call mutates it)
                net = K.hydrate(base_path, cfg, mod, dm)
                t = time.time()
                net = T.finetune_predictor_on_l4(
                    net, dm, lambda_4=lam, epochs=T.FINETUNE_EPOCHS,
                    every_k=(k if schedule == "every_k" else 1),
                    after_k=(k * spe if schedule == "after_k" else 0),
                    l4_kind=loss, imm_features=sp["fixed"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(net.params, f)
                print(f"[{i}/{len(todo)}] fine-tuned {name} in {time.time() - t:.0f}s",
                      flush=True)
        r = full_row(net, dm, sp, ds)
        r.update(loss=loss, phase=phase, schedule=schedule,
                 k=(k if schedule != "always" else 0), lambda_4=lam, model=name)
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / f"{ds}_grid_results.csv", index=False)

    after = K.fingerprint(PROTECTED)
    print(f"\n{len(rows)} cells -> {OUT / f'{ds}_grid_results.csv'}", flush=True)
    print(f"existing result folders unchanged: {before == after}", flush=True)
    assert before == after


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in K.SPEC, "usage: table7_grid.py adult|credit|oulad"
    main(sys.argv[1])
