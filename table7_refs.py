"""Reference and RQ1 models with the CounterNet paper's Table 7 settings.

Completes the Table 7 results end to end (table7_grid.py covers the feasibility baselines and
the regularizer grids):

  Adult   the RQ1 study of thesis_immutability_experiment.py: no constraints, the paper's
          projection of race and gender, projection of all three immutable features, the
          Generator penalty at seven weights, the invariant Predictor (consistency and
          gradient, three weights each), the combination at three weights, and the
          three-feature projection model evaluated without its projection
  Credit  no constraints, the paper's projection of sex, projection of sex and marriage
  OULAD   no constraints, the paper's projection of gender, projection of all three

Same model classes, training and evaluation as the default runs; only the architecture and
learning rate follow table7_check.TABLE7.

    python table7_refs.py

Writes only to trained_models/table7_refs/. A fingerprint of the existing result folders is
taken before and after the run to confirm they are untouched.
"""
import pickle, time
from pathlib import Path

import pandas as pd

import thesis_immutability_experiment as T
import cred_grid as C
import oulad_grid as O
import table7_check as K
from relax.methods.counternet import CounterNet

OUT = Path("trained_models/table7_refs")
PROTECTED = K.PROTECTED + [K.OUT, Path("trained_models/table7_grid")]
for _p in PROTECTED:
    assert OUT.resolve() != _p.resolve() and _p.resolve() not in OUT.resolve().parents

GEN_LAMBDAS = [1.0, 5.0, 20.0, 100.0, 1000.0, 10000.0, 1e6]
INV_LAMBDAS = [0.1, 1.0, 10.0]
COMBO_LAMBDAS = [1.0, 20.0, 1000.0]


def plain(arch):
    return T.CounterNetConfig(**arch), None


def train_or_load(ds, name, cfg, module, dm):
    path = OUT / ds / name / "params.pkl"
    net = CounterNet(cfg) if module is None else CounterNet(cfg, cfnet_module=module)
    if path.exists():
        net.set_data_module(dm)
        net.module = net._init_model(net.config, net.module)
        net.module._data_module = dm
        with open(path, "rb") as f:
            net.params = pickle.load(f)
        net._is_trained = True
        return net
    t = time.time()
    net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
    print(f"  trained {ds}/{name} in {time.time() - t:.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(net.params, f)
    return net


def adult(arch):
    rows = []

    def add(name, label, cfg, module, immutable):
        dm = T.load_dm(immutable)
        net = train_or_load("adult", name, cfg, module, dm)
        r = T.evaluate_variant(net, dm)
        r.update(model=name, label=label)
        rows.append(r)

    cfg = T.ActionabilityConfig(penalized_features=T.IMMUTABLE, lambda_4=0.0, **arch)
    add("actionability_l4_0.0", "no constraints", cfg, T.ActionabilityCounterNetModule(cfg), [])
    add("feasibility_race_gender", "projection of race and gender (CounterNet paper)",
        *plain(arch), ["race", "gender"])
    add("projection", "projection of race, gender, marital status", *plain(arch), T.IMMUTABLE)
    for lam in GEN_LAMBDAS:
        cfg = T.ActionabilityConfig(penalized_features=T.IMMUTABLE, lambda_4=lam, **arch)
        add(f"actionability_l4_{lam}", f"Generator loss, lambda_imm = {lam:g}", cfg,
            T.ActionabilityCounterNetModule(cfg), [])
    for kind in ("consistency", "gradient"):
        for lam in INV_LAMBDAS:
            cfg = T.InvarianceConfig(invariant_features=T.IMMUTABLE, lambda_5=lam,
                                     invariance_kind=kind, **arch)
            add(f"invariance_{kind}_l5_{lam}", f"invariant Predictor, {kind}, lambda_inv = {lam:g}",
                cfg, T.InvariantPredictorCounterNetModule(cfg), [])
    for lam in COMBO_LAMBDAS:
        cfg = T.InvariantActionabilityConfig(
            penalized_features=T.IMMUTABLE, lambda_4=lam, invariant_features=T.IMMUTABLE,
            lambda_5=10.0, invariance_kind="consistency", **arch)
        add(f"combined_consistency_l5_10.0_l4_{lam}",
            f"invariant Predictor (consistency, lambda_inv = 10) + Generator loss, lambda_imm = {lam:g}",
            cfg, T.InvariantActionabilityModule(cfg), [])
    # the three-feature projection model, generating without its projection
    dm_free = T.load_dm([])
    net = train_or_load("adult", "projection", *plain(arch), dm_free)
    r = T.evaluate_variant(net, dm_free)
    r.update(model="projection_removed_at_test", label="projection removed at test time")
    rows.append(r)
    return rows


def references(ds, mod, paper_immutable, arch):
    rows = []
    for name, immutable, label in (
            (f"{ds}_base", [], "no constraints"),
            (f"{ds}_feasibility_paper", paper_immutable,
             f"projection of {', '.join(paper_immutable)} (CounterNet paper)"),
            (f"{ds}_projection", mod.FIXED, f"projection of {', '.join(mod.FIXED)}")):
        dm = mod.load_dm(immutable)
        net = train_or_load(ds, name, *plain(arch), dm)
        r = mod.evaluate(net, dm)
        r.update(model=name, label=label)
        rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = K.fingerprint(PROTECTED)
    print("== adult (RQ1 study) ==", flush=True)
    pd.DataFrame(adult(K.TABLE7["adult"])).to_csv(OUT / "adult_rq1.csv", index=False)
    for ds, mod, paper in (("credit", C, ["SEX"]), ("oulad", O, ["gender"])):
        print(f"== {ds} (references) ==", flush=True)
        pd.DataFrame(references(ds, mod, paper, K.TABLE7[ds])).to_csv(
            OUT / f"{ds}_reference.csv", index=False)
    after = K.fingerprint(PROTECTED)
    print(f"\nexisting result folders unchanged: {before == after}", flush=True)
    assert before == after


if __name__ == "__main__":
    main()
