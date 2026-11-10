"""Robustness check: the selected models retrained with the CounterNet paper's Table 7 settings.

The main results use the CounterNet repository's single default configuration for every
dataset (Encoder [50, 10], Predictor [10], Generator [50, 50], learning rate 0.003). This
script retrains, per dataset, the feasibility baseline and the selected configuration of
regularizers A and B with Guo et al.'s Table 7 settings, everything else unchanged:

  Adult, Credit  learning rate 0.003, Encoder [50, 10], Predictor [10], Generator [50]
  OULAD          learning rate 0.001, Encoder [200, 10], Predictor [10], Generator [200]

and evaluates them next to the cached default-configuration models, which are only read.

Writes only to trained_models/table7_check/. A fingerprint of the existing result folders
is taken before and after the run to confirm they are untouched.
"""
import hashlib, os, pickle, time
from pathlib import Path

import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jrand

import thesis_immutability_experiment as T
import ar_grid as A
import cred_grid as C
import oulad_grid as O
from relax.methods.counternet import CounterNet

OUT = Path("trained_models/table7_check")
PROTECTED = [T.OUT_DIR, Path("trained_models/oulad_table7_test")]
for _p in PROTECTED:
    assert OUT.resolve() != _p.resolve() and _p.resolve() not in OUT.resolve().parents

TABLE7 = {
    "adult": dict(lr=0.003, enc_sizes=[50, 10], pred_sizes=[10], exp_sizes=[50]),
    "credit": dict(lr=0.003, enc_sizes=[50, 10], pred_sizes=[10], exp_sizes=[50]),
    "oulad": dict(lr=0.001, enc_sizes=[200, 10], pred_sizes=[10], exp_sizes=[200]),
}
FAV = {"adult": 1, "credit": 0, "oulad": 1}  # favourable predicted class

# (phase, loss, schedule, k, lambda_4, cached default-configuration model)
SPEC = {
    "adult": dict(
        load=lambda: T.load_dm(T.IMMUTABLE), proj=A._proj_kwargs(), fixed=T.IMMUTABLE,
        evaluate=T.evaluate_variant, out=T.OUT_DIR,
        baseline="reachability_projection_fixed3_age_nondec_no_unknown",
        A=("two-phase", "proximity", "every_k", 5, 0.5,
           "actionability_recourse_finetune_everyk5_l4_0.5_20ep"),
        B=("joint", "gradient", "every_k", 50, 5.0,
           "actionability_recourse_gradpen_every_k_50_l4_5.0")),
    "credit": dict(
        load=lambda: C.load_dm(C.FIXED), proj=C.proj_kwargs(), fixed=C.FIXED,
        evaluate=C.evaluate, out=C.OUT, baseline="cred_reachability_baseline",
        A=("joint", "proximity", "after_k", 10, 0.5, "cred_predprox_after_k_1760_l4_0.5"),
        B=("joint", "gradient", "every_k", 10, 10.0, "cred_gradpen_every_k_10_l4_10.0")),
    "oulad": dict(
        load=lambda: O.load_dm(O.FIXED), proj=O.proj_kwargs(), fixed=O.FIXED,
        evaluate=O.evaluate, out=O.OUT, baseline="oulad_reachability_baseline",
        A=("joint", "proximity", "after_k", 20, 10.0, "oulad_predprox_after_k_3820_l4_10.0"),
        B=("joint", "gradient", "after_k", 10, 0.5, "oulad_gradpen_after_k_1910_l4_0.5")),
}


def fingerprint(paths):
    h = hashlib.sha256()
    for root in paths:
        for dirpath, _, files in sorted(os.walk(root)):
            for f in sorted(files):
                p = Path(dirpath) / f
                st = p.stat()
                h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def baseline_cfg(sp, arch):
    cfg = T.ReachabilityProjectionConfig(**sp["proj"], **arch)
    return cfg, T.ReachabilityProjectionModule(cfg)


def joint_cfg(sp, loss, schedule, k, lam, spe, arch):
    # after_k is given in epochs; the schedule gate counts training steps
    kk = k * spe if schedule == "after_k" else k
    if loss == "proximity":
        cfg = T.ScheduledPredictorProximityConfig(
            **sp["proj"], lambda_4=lam, l4_schedule=schedule, l4_k=kk, **arch)
        return cfg, T.ScheduledPredictorProximityModule(cfg)
    cfg = T.ScheduledImmutableGradientConfig(
        **sp["proj"], invariant_features=sp["fixed"], lambda_4=lam,
        l4_schedule=schedule, l4_k=kk, **arch)
    return cfg, T.ScheduledImmutableGradientModule(cfg)


def hydrate(path, cfg, module, dm):
    net = CounterNet(cfg, cfnet_module=module)
    net.set_data_module(dm)
    net.module = net._init_model(net.config, net.module)
    net.module._data_module = dm
    with open(path, "rb") as f:
        net.params = pickle.load(f)
    net._is_trained = True
    return net


def train(cfg, module, dm, name, ds):
    path = OUT / ds / name / "params.pkl"
    if path.exists():
        return hydrate(path, cfg, module, dm)
    net = CounterNet(cfg, cfnet_module=module)
    t = time.time()
    net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
    print(f"  trained {ds}/{name} in {time.time() - t:.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(net.params, f)
    return net


def recourse(net, dm, fav):
    X = jnp.asarray(dm["test"][0])
    cf = net.module.generate_cf(net.params, jrand.PRNGKey(0), X)
    yx = np.asarray(net.pred_fn(X)).argmax(-1)
    yc = np.asarray(net.pred_fn(cf)).argmax(-1)
    l1 = np.abs(np.asarray(cf) - np.asarray(X)).sum(1)
    rej = yx != fav
    ok = rej & (yc != yx)
    return dict(recourse_validity=float(ok.sum() / rej.sum()),
                recourse_cost=float(l1[ok].mean()))


def row(net, dm, sp, ds, model, setting):
    r = sp["evaluate"](net, dm)
    r.update(recourse(net, dm, FAV[ds]))
    keep = ["acc", "validity", "recourse_validity", "proximity", "recourse_cost", "viol_any"]
    return dict(dataset=ds, model=model, setting=setting, **{k: r[k] for k in keep})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = fingerprint(PROTECTED)
    rows = []
    for ds, sp in SPEC.items():
        print(f"== {ds} ==", flush=True)
        dm = sp["load"]()
        spe = int(np.ceil(len(dm["train"][0]) / T.BATCH_SIZE))

        # default configuration: cached models, read only
        cfg, mod = baseline_cfg(sp, {})
        net = hydrate(sp["out"] / sp["baseline"] / "params.pkl", cfg, mod, dm)
        rows.append(row(net, dm, sp, ds, "feasibility baseline", "default"))
        for reg in ("A", "B"):
            phase, loss, schedule, k, lam, name = sp[reg]
            cfg, mod = (baseline_cfg(sp, {}) if phase == "two-phase"
                        else joint_cfg(sp, loss, schedule, k, lam, spe, {}))
            net = hydrate(sp["out"] / name / "params.pkl", cfg, mod, dm)
            rows.append(row(net, dm, sp, ds, f"regularizer {reg}", "default"))

        # Table 7 settings: trained here
        arch = TABLE7[ds]
        cfg, mod = baseline_cfg(sp, arch)
        base = train(cfg, mod, dm, "feasibility_baseline", ds)
        rows.append(row(base, dm, sp, ds, "feasibility baseline", "table7"))
        for reg in ("A", "B"):
            phase, loss, schedule, k, lam, _ = sp[reg]
            if phase == "two-phase":
                name = f"regularizer_{reg}"
                path = OUT / ds / name / "params.pkl"
                cfg, mod = baseline_cfg(sp, arch)
                if path.exists():
                    net = hydrate(path, cfg, mod, dm)
                else:
                    # fine-tune a fresh copy of the Table 7 baseline (the call mutates it)
                    net = hydrate(OUT / ds / "feasibility_baseline" / "params.pkl",
                                  cfg, mod, dm)
                    t = time.time()
                    net = T.finetune_predictor_on_l4(
                        net, dm, lambda_4=lam, epochs=T.FINETUNE_EPOCHS,
                        every_k=(k if schedule == "every_k" else 1),
                        after_k=(k * spe if schedule == "after_k" else 0),
                        l4_kind=loss, imm_features=sp["fixed"])
                    print(f"  fine-tuned {ds}/{name} in {time.time() - t:.0f}s", flush=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as f:
                        pickle.dump(net.params, f)
            else:
                cfg, mod = joint_cfg(sp, loss, schedule, k, lam, spe, arch)
                net = train(cfg, mod, dm, f"regularizer_{reg}", ds)
            rows.append(row(net, dm, sp, ds, f"regularizer {reg}", "table7"))
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    print(df.to_string(index=False), flush=True)
    after = fingerprint(PROTECTED)
    print(f"\nexisting result folders unchanged: {before == after}", flush=True)
    assert before == after


if __name__ == "__main__":
    main()
