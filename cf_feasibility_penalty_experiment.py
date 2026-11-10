"""Extend the released C1/C2 soft objectives to the CounterNet feasibility set.

No policy projection is allowed in training or evaluation. Baselines are the
completed original runs, reused read-only. This is a fixed-weight experiment,
not a claim that the original paper proposed penalties for these extra features.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import csv
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

import cf_feasibility_experiment as base
import cf_feasibility_c2_experiment as c2

OUT = base.ROOT / "trained_models/cf_feasibility_penalties"
SEEDS = (42, 43, 44)
VARIANT = "loss_only"


def settings(setting):
    if setting == "c1":
        return base.Config(), base.DEFAULT_OUT, base.objective, base.candidate_metrics
    if setting == "c2":
        return c2.C2Config(), c2.OUT, c2.c2_objective, c2.c2_metrics
    raise ValueError(setting)


def added_penalties(raw, x, policy):
    """One hinge per inequality, averaged over draws/people, summed over rules.

    An equality r_j=x_j is two inequalities, hence |r_j-x_j|. Prohibiting an
    output category is r_j<=0, hence relu(r_j) for sigmoid scores. The native
    objective already contains the age hinge, so it is not counted again here.
    """
    terms = {}
    for name in policy.fixed_names:
        f = policy.by_name[name]
        delta = raw[..., f["start"]:f["end"]] - x[..., f["start"]:f["end"]]
        terms["penalty_" + name] = (F.relu(delta) + F.relu(-delta)).sum(-1).mean()
    for name, index in policy.banned.items():
        terms["penalty_" + name + "_unknown"] = F.relu(raw[..., index]).mean()
    return sum(terms.values()), terms


def penalty_objective(raw, mean, variance, x, target, classifier, policy, cfg, extended):
    assert not extended, "Loss-only experiments cannot use the projected variant"
    native = c2.c2_objective if isinstance(cfg, c2.C2Config) else base.objective
    loss, terms = native(raw, mean, variance, x, target, classifier, policy, cfg, False)
    added, details = added_penalties(raw, x, policy)
    total = loss + cfg.age_weight * added
    terms.update(details)
    terms.update(native_total=loss, added_penalty=added,
                 added_weighted_penalty=cfg.age_weight * added, total=total)
    return total, terms


def forbidden_projection(*args, **kwargs):
    raise AssertionError("Policy projection is forbidden in loss-only experiments")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, setting):
    cfg, original, _, _ = settings(setting)
    out = root / setting
    out.mkdir(parents=True, exist_ok=True)
    for name in ("adult_inputs.npz", "data_manifest.json", "classifier.pt", "classifier_results.json", "environment.txt", "upstream_sources.json"):
        c2.copy_identical(original / name, out / name)
    protocol = dict(
        experiment="Additional feasibility constraints through soft losses only",
        setting=setting, config=asdict(cfg), seeds=list(SEEDS), variant=VARIANT,
        baseline_directory=str(original.relative_to(base.ROOT)),
        inherited_protocol=json.loads((original / "protocol.json").read_text()),
        projection=False,
        added_loss="lambda_F * mean_draws_people(sum_fixed_coordinates(abs(r_j-x_j)) + r_workclass_unknown + r_occupation_unknown)",
        fixed_features=list(base.AdultPolicy.fixed_names),
        forbidden_generated_categories=base.AdultPolicy.banned_names,
        age="Original raw age-decrease hinge retained once; no age floor.",
        education="Original released C2 linear education-score term retained unchanged." if setting == "c2" else "No education restriction.",
        weight=cfg.age_weight,
        weight_choice="Reuse the setting's published feasibility coefficient (C1=192, C2=87). Sum all encoded constraint hinges; no dimension normalization, additional coefficient, sweep or tuning.",
        weight_scope="More constraints increase the possible total penalty. This is one specified soft-constraint extension, not an optimization over penalty weights.",
        decoding="Common categorical argmax and numerical [0,1] bounds only. No fixed-feature copying, age floor, forbidden-category masking or allowed-category renormalization.",
        evaluation="All 10 draws for every one of 8141 test observations, both prediction directions, three matched Generator seeds. Report decoded validity, L1 and policy violations; separately check the full C2 conditional rule.",
        baseline_reuse="Read-only reuse of the original checkpoints/evaluations; same classifier, inputs, train/test indices, Generator initialization, data order, training/evaluation seeds and training budget.",
        source_revision=base.UPSTREAM_COMMIT,
        implementation_sha256={name:fingerprint(base.ROOT/name) for name in (
            "cf_feasibility_experiment.py", "cf_feasibility_c2_experiment.py", "cf_feasibility_penalty_experiment.py")})
    path = out / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Protocol changed: use a fresh output directory"
    else:
        base.save_json(path, protocol)
    return out, cfg


def aggregate(root):
    rows, runs, summary, paired = [], [], {}, {}
    for setting in ("c1", "c2"):
        cfg, original, _, _ = settings(setting)
        for variant, directory in (("original", original), (VARIANT, root / setting)):
            selected = []
            for seed in SEEDS:
                path = directory / f"{variant}_seed{seed}"
                run = json.loads((path / "evaluation.json").read_text())
                run = {**run, "setting":setting, "artifact_directory":str(path.relative_to(base.ROOT)),
                       "evaluation_sha256":fingerprint(path/"evaluation.json")}
                runs.append(run)
                for cohort, metrics in run["cohorts"]["decoded"].items():
                    row = dict(setting=setting, variant=variant, seed=seed, cohort=cohort, **metrics)
                    rows.append(row)
                    selected.append(row)
            for cohort in ("all", "low_to_high", "high_to_low"):
                members = [r for r in selected if r["cohort"] == cohort]
                values = {}
                for metric in members[0]:
                    if metric in ("setting", "variant", "seed", "cohort"):
                        continue
                    samples = [r[metric] for r in members if r[metric] is not None]
                    values[metric] = dict(mean=float(np.mean(samples)), sd=float(np.std(samples, ddof=1))) if samples else None
                summary[f"{setting}/{variant}/{cohort}"] = dict(seeds=list(SEEDS), metrics=values)
        for cohort in ("all", "low_to_high", "high_to_low"):
            pair = {v:{r["seed"]:r for r in rows if r["setting"] == setting and r["variant"] == v and r["cohort"] == cohort} for v in ("original", VARIANT)}
            metrics = ("valid", "proximity", "feasible", "valid_feasible") + (("c2_compliant", "valid_feasible_c2") if setting == "c2" else ())
            diffs = {k:[pair[VARIANT][s][k]-pair["original"][s][k] for s in SEEDS] for k in metrics}
            paired[f"{setting}/{cohort}"] = {k:dict(differences=v, mean=float(np.mean(v)), sd=float(np.std(v,ddof=1))) for k,v in diffs.items()}
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader(); writer.writerows(rows)
    base.save_json(root/"summary.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocols={s:json.loads((root/s/"protocol.json").read_text()) for s in ("c1","c2")},
        classifier=json.loads((root/"c1/classifier_results.json").read_text()),
        summary=summary, paired=paired, runs=runs))
    print("Wrote loss-only summary and results CSV", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "prepare", "train", "aggregate"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--setting", choices=["c1", "c2"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "aggregate":
        aggregate(args.out)
        return
    for setting in ([args.setting] if args.setting else ["c1", "c2"]):
        out, cfg = prepare(args.out, setting)
        if args.command == "prepare":
            continue
        for seed in ([args.seed] if args.seed else SEEDS):
            # Enforce the user's no-projection requirement throughout both loops.
            with patch.object(base.AdultPolicy, "project", forbidden_projection):
                base.train_generator(out, cfg, VARIANT, seed, objective_fn=penalty_objective)
                base.evaluate_generator(out, cfg, VARIANT, seed, metrics_fn=settings(setting)[3])
    if args.command == "all" and not args.setting and args.seed is None:
        aggregate(args.out)


if __name__ == "__main__":
    main()
