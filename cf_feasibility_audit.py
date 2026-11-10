"""Independently audit saved CF results and construct feasible linear witnesses.

No training, tuning or selection. All checks use the completed fixed experiment.
"""
from pathlib import Path
import json
import numpy as np
import torch
from cf_feasibility_experiment import DEFAULT_OUT, Config, BlackBox, state_hash


def audit(out=DEFAULT_OUT):
    torch.set_num_threads(1)
    data = np.load(out / "adult_inputs.npz")
    meta = json.loads((out / "data_manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    features = meta["features"]
    by_name = {f["name"]: f for f in features}
    x = data["x"][data["test_indices"]]
    checkpoint = torch.load(out / "classifier.pt", map_location="cpu", weights_only=True)
    classifier = BlackBox().eval()
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.requires_grad_(False)
    with torch.no_grad():
        factual = classifier(torch.from_numpy(x)).argmax(-1).numpy()
    classifier_hash = state_hash(classifier)
    initial = {}
    checked = 0
    for run in summary["runs"]:
        path = out / f'{run["variant"]}_seed{run["seed"]}'
        trained = torch.load(path / "generator.pt", map_location="cpu", weights_only=True)
        assert trained["classifier_sha256"] == classifier_hash
        key = run["seed"]
        if key in initial:
            assert initial[key] == trained["initial_sha256"]
        initial[key] = trained["initial_sha256"]
        cf_file = np.load(path / "test_candidates.npz")
        np.testing.assert_array_equal(cf_file["test_indices"], data["test_indices"])
        np.testing.assert_array_equal(cf_file["factual_prediction"], factual)
        for mode, cohorts in run["cohorts"].items():
            cf = cf_file["cf_" + mode]
            assert cf.shape == (10, 8141, 29) and np.isfinite(cf).all()
            with torch.no_grad():
                pred = classifier(torch.from_numpy(cf)).argmax(-1).numpy()
            np.testing.assert_array_equal(pred, cf_file["prediction_" + mode])
            valid = pred != factual[None]
            distance = np.abs(cf - x[None]).sum(-1)
            violation = np.zeros(valid.shape, dtype=bool)
            for f in features:
                s, e = f["start"], f["end"]
                block = cf[..., s:e]
                if f["categorical"]:
                    assert ((block == 0) | (block == 1)).all()
                    assert (block.sum(-1) == 1).all()
                else:
                    assert (block >= 0).all() and (block <= 1).all()
                if f["name"] in ("race", "gender", "marital_status"):
                    violation |= np.any(block != x[None, :, s:e], axis=-1)
                if f["name"] in ("workclass", "occupation"):
                    j = f["categories"].index("Other/Unknown")
                    violation |= block[..., j] == 1
            violation |= cf[..., 0] < x[None, :, 0] - 1e-6
            for cohort, expected in cohorts.items():
                mask = np.ones(len(x), bool) if cohort == "all" else factual == (0 if cohort == "low_to_high" else 1)
                checks = dict(valid=valid[:, mask].mean(), proximity=distance[:, mask].mean(),
                              feasible=(~violation[:, mask]).mean(),
                              valid_feasible=(valid[:, mask] & ~violation[:, mask]).mean())
                for metric, value in checks.items():
                    np.testing.assert_allclose(value, expected[metric], rtol=1e-6, atol=1e-6)
            checked += cf.shape[0] * cf.shape[1]
    # With the source's two affine layers and no nonlinearity, the margin is
    # affine. Extremize each permitted block independently to get an exact
    # decision-attainability witness (no proximity budget, integer rounding or SCM).
    layer1, layer2 = classifier.predict_net
    w = ((layer2.weight[1] - layer2.weight[0]) @ layer1.weight).detach().numpy()
    witness = x.copy()
    direction = np.where(factual == 0, 1.0, -1.0)
    for f in features:
        name, s, e = f["name"], f["start"], f["end"]
        if name in ("race", "gender", "marital_status"):
            continue
        if f["categorical"]:
            scores = direction[:, None] * w[None, s:e]
            if name in ("workclass", "occupation"):
                scores[:, f["categories"].index("Other/Unknown")] = -np.inf
            witness[:, s:e] = np.eye(e-s, dtype=np.float32)[scores.argmax(-1)]
        else:
            lower = x[:, s] if name == "age" else np.zeros(len(x))
            witness[:, s] = np.where(direction * w[s] >= 0, 1.0, lower)
    with torch.no_grad():
        witness_prediction = classifier(torch.from_numpy(witness)).argmax(-1).numpy()
    attainable = witness_prediction != factual
    assert (witness[:, 0] >= x[:, 0]).all()
    for name in ("race", "gender", "marital_status"):
        f = by_name[name];np.testing.assert_array_equal(witness[:, f["start"]:f["end"]], x[:, f["start"]:f["end"]])
    for name in ("workclass", "occupation"):
        f = by_name[name];assert (witness[:, f["start"]+f["categories"].index("Other/Unknown")] == 0).all()
    reachability = {}
    for name, mask in [("all", np.ones(len(x), bool)), ("low_to_high", factual == 0), ("high_to_low", factual == 1)]:
        reachability[name] = dict(n_people=int(mask.sum()), attainable=int(attainable[mask].sum()), rate=float(attainable[mask].mean()))
    np.savez_compressed(out / "linear_feasibility_witnesses.npz", test_indices=data["test_indices"],
                        witnesses=witness, factual_prediction=factual, witness_prediction=witness_prediction)
    result = dict(passed=True, trained_runs=len(summary["runs"]), audited_decoded_candidates=checked,
                  classifier_sha256=classifier_hash, paired_initializations=initial,
                  decision_attainability=reachability,
                  interpretation="Feasible class-flip witnesses for this linear classifier, with no distance budget; not a causal or low-cost guarantee.")
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
