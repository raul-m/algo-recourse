"""Small test: regularizer A with an L1 distance instead of the squared distance.

Regularizer A adds L4 = 0.5 * mean (x - x'')^2 on the projected counterfactual to the
Predictor's objective. This script swaps only that distance for L4 = mean |x - x''|,
keeping lambda_4, the schedule, the projection and stage 2 (L2, L3) unchanged, and runs
the three best regularizer-A configurations per dataset (validity within 0.2 pp and
accuracy within 0.5 pp of the feasibility baseline, lowest proximity).

  Adult   the best cells are two-phase. Fine-tuning is seeded, so each L1 cell starts
          from the cached feasibility baseline and is compared with the cached squared
          cell, which reproduces bit for bit. One run each.
  Credit, the best cells use joint training, which is not reproducible run to run
  OULAD   (unseeded batch shuffle in the relax trainer). Each configuration is trained
          REPEATS times with L1 and REPEATS times with the squared form.

Nothing under trained_models/immutability_experiment is written. Results go to
trained_models/l1_regularizer_test/.
"""
import sys
import pickle
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
import pandas as pd

import thesis_immutability_experiment as T
import ar_grid as A
import cred_grid as C
import oulad_grid as O
from relax.legacy.utils import grad_update
from relax.methods.counternet import CounterNet, partition_trainable_params

OUT = Path("trained_models/l1_regularizer_test")
assert T.OUT_DIR.resolve() not in OUT.resolve().parents and OUT.resolve() != T.OUT_DIR.resolve()
REPEATS = 3

# (dataset, phase, schedule, k, lambda_4, cached squared-form model name)
CONFIGS = [
    ("adult", "two-phase", "every_k", 5, 0.5, "actionability_recourse_finetune_everyk5_l4_0.5_20ep"),
    ("adult", "two-phase", "every_k", 50, 5.0, "actionability_recourse_finetune_everyk50_l4_5.0_20ep"),
    ("adult", "two-phase", "every_k", 20, 1.0, "actionability_recourse_finetune_everyk20_l4_1.0_20ep"),
    ("credit", "joint", "after_k", 10, 0.5, "cred_predprox_after_k_1760_l4_0.5"),
    ("credit", "joint", "every_k", 5, 0.5, "cred_predprox_every_k_5_l4_0.5"),
    ("credit", "joint", "always", 0, 0.5, "cred_predprox_l4_0.5"),
    ("oulad", "joint", "after_k", 20, 10.0, "oulad_predprox_after_k_3820_l4_10.0"),
    ("oulad", "joint", "after_k", 20, 5.0, "oulad_predprox_after_k_3820_l4_5.0"),
    ("oulad", "joint", "after_k", 10, 5.0, "oulad_predprox_after_k_1910_l4_5.0"),
]


def l1_distance(x, cf):
    return jnp.mean(jnp.abs(x - cf))


def _l4_candidate(module, params, rng_key, x, is_training):
    """Projected counterfactual with the Generator's weights frozen (as in regularizer A)."""
    explainer_params, backbone_params = partition_trainable_params(
        params, trainable_name="counter_net_model/Explainer")
    frozen = hk.data_structures.merge(
        jax.lax.stop_gradient(explainer_params), backbone_params)
    _, cf = module.net.apply(frozen, rng_key, x, is_training=is_training)
    return module.apply_reachability_projection(x, cf, hard=not is_training)


class L1PredictorProximityModule(T.PredictorProximityReachabilityModule):
    """Regularizer A, schedule 'always', with L4 = mean |x - x''|."""

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_4 = l1_distance(x, _l4_candidate(self, params, rng_key, x, is_training))
        return (self.configs.lambda_1 * self.loss_fn_1(y_pred, y)
                + self.configs.lambda_4 * loss_4)


class L1ScheduledPredictorProximityModule(T.ScheduledPredictorProximityModule):
    """Regularizer A, schedules 'every_k' / 'after_k', with L4 = mean |x - x''|."""

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, l4_gate, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_4 = l1_distance(x, _l4_candidate(self, params, rng_key, x, is_training))
        return (self.configs.lambda_1 * self.loss_fn_1(y_pred, y)
                + l4_gate * self.configs.lambda_4 * loss_4)


def finetune_l1(cfnet, dm, lambda_4, epochs, every_k=1, after_k=0, seed=T.TRAIN_SEED):
    """Two-phase fine-tuning of T.finetune_predictor_on_l4 (proximity kind), with the same
    seed, batch order and gating, but L4 = mean |x - x''|."""
    module, params = cfnet.module, cfnet.params
    _ = module.fixed_mask, module.nondecreasing_mask, module.disallowed_category_mask
    opt = optax.adam(learning_rate=module.configs.lr)
    backbone, generator = hk.data_structures.partition(
        lambda m, n, v: "counter_net_model/Explainer" not in m, params)
    opt_state = opt.init(backbone)

    def loss_fn(backbone_params, generator_params, rng_key, batch, l4_gate):
        x, y = batch
        merged = hk.data_structures.merge(backbone_params, generator_params)
        y_pred, _ = module.net.apply(merged, rng_key, x, is_training=True)
        frozen = hk.data_structures.merge(
            backbone_params, jax.lax.stop_gradient(generator_params))
        _, cf = module.net.apply(frozen, rng_key, x, is_training=True)
        cf = module.apply_reachability_projection(x, cf, hard=False)
        return (module.configs.lambda_1 * module.loss_fn_1(y_pred, y)
                + l4_gate * lambda_4 * l1_distance(x, cf))

    @jax.jit
    def step(backbone_params, generator_params, opt_state, rng_key, batch, l4_gate):
        grads = jax.grad(loss_fn)(backbone_params, generator_params, rng_key, batch, l4_gate)
        return grad_update(grads, backbone_params, opt_state, opt)

    xs, ys = dm["train"]
    n, key, step_i = xs.shape[0], jrand.PRNGKey(seed), 0
    for _ in range(epochs):
        key, shuffle_key = jrand.split(key)
        order = jrand.permutation(shuffle_key, n)
        for start in range(0, n, T.BATCH_SIZE):
            idx = order[start:start + T.BATCH_SIZE]
            key, step_key = jrand.split(key)
            gate = jnp.asarray(1.0 if (step_i % every_k == 0 and step_i >= after_k) else 0.0)
            backbone, opt_state = step(backbone, generator, opt_state, step_key,
                                       (xs[idx], ys[idx]), gate)
            step_i += 1
    cfnet.params = hk.data_structures.merge(backbone, generator)
    return cfnet


def setup(ds):
    if ds == "adult":
        return dict(G=A, load=lambda: T.load_dm(T.IMMUTABLE), proj=A._proj_kwargs(),
                    out=A.OUT, base="reachability_projection_fixed3_age_nondec_no_unknown",
                    evaluate=T.evaluate_variant, fixed=T.IMMUTABLE)
    G = {"credit": C, "oulad": O}[ds]
    return dict(G=G, load=lambda: G.load_dm(G.FIXED), proj=G.proj_kwargs(), out=G.OUT,
                base=G.BASELINE_NAME, evaluate=G.evaluate, fixed=G.FIXED)


def hydrate(s, name, dm):
    cfg = T.ReachabilityProjectionConfig(**s["proj"])
    net = CounterNet(cfg, cfnet_module=T.ReachabilityProjectionModule(cfg))
    net.set_data_module(dm)
    net.module = net._init_model(net.config, net.module)
    net.module._data_module = dm
    net.params = pickle.load(open(s["out"] / name / "params.pkl", "rb"))
    net._is_trained = True
    m = net.module
    _ = m.fixed_mask, m.nondecreasing_mask, m.disallowed_category_mask, m.ordinal_blocks
    return net


def switches_per_person(net, dm, fixed):
    """Mean number of mutable categorical features whose category changes."""
    xs, _ = dm["test"]
    cf = np.asarray(net.module.generate_cf(net.params, jrand.PRNGKey(0), xs))
    xs = np.asarray(xs)
    n = np.zeros(len(xs))
    for f, (s, e) in zip(list(dm.features), dm.features.feature_indices):
        if f.is_categorical and f.name not in fixed:
            n += np.abs(cf[:, s:e] - xs[:, s:e]).sum(1) > 1e-6
    return float(n.mean())


def score(s, net, row):
    dm = s["load"]()
    net.module._data_module = dm
    r = s["evaluate"](net, dm)
    row.update(acc=r["acc"], validity=r["validity"], proximity=r["proximity"],
               viol_any=r["viol_any"], switches=switches_per_person(net, s["load"](), s["fixed"]))
    return row


def run(ds_filter=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, phase, sched, k, lam, cached in CONFIGS:
        if ds_filter and ds != ds_filter:
            continue
        s = setup(ds)
        G, dm = s["G"], s["load"]()
        spe = int(np.ceil(len(dm["train"][0]) / T.BATCH_SIZE))
        tag = f"{ds}_{phase}_{sched}_k{k}_l4_{lam}"
        base_row = dict(dataset=ds, phase=phase, schedule=sched, k=k, lambda_4=lam)
        rows.append(score(s, hydrate(s, cached, s["load"]()),
                          dict(base_row, form="squared", run="cached")))
        if phase == "two-phase":
            net = finetune_l1(hydrate(s, s["base"], dm), dm, lam, T.FINETUNE_EPOCHS,
                              every_k=(k if sched == "every_k" else 1),
                              after_k=(k * spe if sched == "after_k" else 0))
            (OUT / f"{tag}_l1").mkdir(exist_ok=True)
            pickle.dump(net.params, open(OUT / f"{tag}_l1" / "params.pkl", "wb"))
            rows.append(score(s, net, dict(base_row, form="L1", run="1")))
        else:
            for rep in range(1, REPEATS + 1):
                for form in ("squared", "L1"):
                    cfg, sq_module = G.joint_model("proximity", sched, k, lam, spe)
                    if form == "squared":
                        module = sq_module
                    elif sched == "always":
                        module = L1PredictorProximityModule(cfg)
                    else:
                        module = L1ScheduledPredictorProximityModule(cfg)
                    net = CounterNet(cfg, cfnet_module=module)
                    net.train(s["load"](), batch_size=T.BATCH_SIZE, epochs=T.EPOCHS,
                              seed=T.TRAIN_SEED)
                    d = OUT / f"{tag}_{form}_run{rep}"
                    d.mkdir(exist_ok=True)
                    pickle.dump(net.params, open(d / "params.pkl", "wb"))
                    rows.append(score(s, net, dict(base_row, form=form, run=str(rep))))
        last = rows[-1]
        print(f"done {tag}: last {last['form']} run {last['run']} acc {last['acc']:.4f} "
              f"val {last['validity']:.4f} prox {last['proximity']:.3f}", flush=True)
        name = f"results_{ds_filter}.csv" if ds_filter else "results.csv"
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
