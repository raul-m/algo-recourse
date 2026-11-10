"""Adult C2 Model-approx: pinned source age/education objective, +/- projection.

Reuses the C1 data, frozen classifier, architecture and training/evaluation loops.
Results go to a separate directory and do not replace the completed C1 runs.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

import cf_feasibility_experiment as base

OUT = base.ROOT / "trained_models/cf_feasibility_c2"
SOURCE_RANK = {"HS-grad": 0, "School": 0, "Bachelors": 1, "Assoc": 1,
               "Some-college": 1, "Masters": 2, "Prof-school": 2, "Doctorate": 3}
PAPER_RANK = {**SOURCE_RANK, "Prof-school": 3}


@dataclass(frozen=True)
class C2Config(base.Config):
    margin: float = 0.344
    validity_weight: float = 76.0
    age_weight: float = 87.0  # Source constraint_reg multiplies BOTH terms.
    education_coefficient: float = 0.05


def education_score(values, policy, ranks=SOURCE_RANK):
    f = policy.by_name["education"]
    weights = values.new_tensor([ranks[name] for name in f["categories"]])
    return (values[..., f["start"]:f["end"]] * weights).sum(-1)


def c2_objective(raw, mean, variance, x, target, classifier, policy, cfg, extended):
    loss, terms = base.objective(raw, mean, variance, x, target, classifier, policy, cfg, extended)
    # Exact released code, NOT the education hinge described in the paper.
    # Keep this on raw decoder output in both arms, just like the native age term.
    education_term = -cfg.education_coefficient * (education_score(raw, policy) - education_score(x, policy)).mean()
    loss = loss + cfg.age_weight * education_term
    terms.update(total=loss, education_term=education_term,
                 constraint_term=terms["age_penalty"] + education_term)
    return loss, terms


def c2_metrics(cfs, x, predicted, cf_predicted, policy):
    metrics = base.candidate_metrics(cfs, x, predicted, cf_predicted, policy)
    f = policy.by_name["education"]
    index = cfs[..., f["start"]:f["end"]].argmax(-1)
    factual_index = x[..., f["start"]:f["end"]].argmax(-1)[None]
    age_delta = cfs[..., policy.age_index] - x[None, :, policy.age_index]
    for ranks, suffix in [(SOURCE_RANK, ""), (PAPER_RANK, "_paper_rank")]:
        scores = np.array([ranks[name] for name in f["categories"]])
        delta = scores[index] - scores[factual_index]
        # Match the repository's decoded rank comparisons and zero offset;
        # use all candidates in both directions, not only predicted class 1.
        compliant = ((delta == 0) & (age_delta >= 0)) | ((delta > 0) & (age_delta > 0))
        metrics["c2_compliant" + suffix] = compliant
        metrics["valid_c2" + suffix] = metrics["valid"] & compliant
        metrics["feasible_c2" + suffix] = metrics["feasible"] & compliant
        metrics["valid_feasible_c2" + suffix] = metrics["valid_feasible"] & compliant
        if not suffix:
            metrics["c2_education_decrease"] = delta < 0
            metrics["c2_higher_education_without_older_age"] = (delta > 0) & (age_delta <= 0)
            metrics["c2_same_education_with_younger_age"] = (delta == 0) & (age_delta < 0)
    return metrics


def copy_identical(source, destination):
    if destination.exists():
        assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest(), destination
    else:
        shutil.copy2(source, destination)


def prepare(out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("adult_inputs.npz", "data_manifest.json", "classifier.pt", "classifier_results.json", "environment.txt"):
        copy_identical(base.DEFAULT_OUT / name, out / name)
    (out / "upstream").mkdir(exist_ok=True)
    # C2-specific sources are installed here from the inspected pinned revision.
    for name in ("unary-ed-const-generative-cf.py", "evaluation_functions.py"):
        if not (out / "upstream" / name).exists():
            raise FileNotFoundError(f"Missing pinned reference source: {name}")
    for source in (base.DEFAULT_OUT / "upstream").glob("*.py"):
        copy_identical(source, out / "upstream" / source.name)
    inherited = json.loads((base.DEFAULT_OUT / "protocol.json").read_text())
    protocol = {**inherited, "config": asdict(cfg), "constraint_setting": "C2 / Adult-Age-Ed",
        "tuning": "None. Fixed published Adult-Age-Ed settings: margin .344, validity 76, feasibility 87; final epoch only.",
        "original": "Released C2 VAE loss: base loss + 87*(age hinge - .05*mean(score(raw)-score(x))).",
        "extended": "Same C2 loss including raw age/education terms, plus the same six-rule projection used in C1.",
        "education_rank_source": SOURCE_RANK, "education_rank_paper": PAPER_RANK,
        "c2_evaluation": "Full conditional age/education rule on decoded candidates, repository ranks; paper-rank sensitivity also reported. No rounding; age > is strict; all candidates and both directions counted.",
        "projection_scope": "Guarantees the existing six-rule policy. Does not force education nondecrease or strictly older age when education increases. C2 compliance measured separately.",
        "source_paper_differences_c2": ["Released education term is linear, not the hinge described in Appendix C.1.6.", "Released training/evaluation rank Prof-school=2; appendix lists 3."],
        "comparison_scope": "Matched original/extended C2; same classifier, data, seeds, epochs and projection as C1. C1/C2 use different published loss weights, so cross-setting differences are not an isolated education-term ablation.",
        "software": dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform())}
    path = out / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text())["config"] == asdict(cfg), "Use a fresh output for new settings"
    base.save_json(path, protocol)
    source_manifest = json.loads((base.DEFAULT_OUT / "upstream_sources.json").read_text())
    source_manifest["files"] += [dict(file=name, repository_path=repo,
        sha256=hashlib.sha256((out / "upstream" / name).read_bytes()).hexdigest()) for name, repo in [
        ("unary-ed-const-generative-cf.py", "generativecf/unary-ed-const-generative-cf.py"),
        ("evaluation_functions.py", "generativecf/scripts/evaluation_functions.py")]]
    base.save_json(out / "upstream_sources.json", source_manifest)
    assert base.state_hash(base.frozen_classifier(out, cfg)) == inherited_classifier_hash()


def inherited_classifier_hash():
    return json.loads((base.DEFAULT_OUT / "classifier_results.json").read_text())["state_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "aggregate", "all"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--variant", choices=["original", "extended"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = C2Config()
    prepare(args.out, cfg)
    pairs = [(s, v) for s in args.seeds for v in ("original", "extended")] if args.command == "all" else [(args.seed, args.variant)]
    for seed, variant in pairs:
        if args.command in ("all", "train"):
            base.train_generator(args.out, cfg, variant, seed, objective_fn=c2_objective)
        if args.command in ("all", "train", "evaluate"):
            base.evaluate_generator(args.out, cfg, variant, seed, metrics_fn=c2_metrics)
    if args.command in ("all", "aggregate"):
        base.aggregate(args.out)


if __name__ == "__main__":
    main()
