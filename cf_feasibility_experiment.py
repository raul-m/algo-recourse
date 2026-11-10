"""Paired Model-approx CF (Adult C1) versus its projected extension.

Method attribution: Mahajan, Tan & Sharma, arXiv:1912.03277.
Reference implementation: https://github.com/divyat09/cf-feasibility
Pinned revision: db715fa107cc1ca328ae6d7863de8a752ca13bdc.

This is an adaptation to the saved CounterNet Adult protocol, not a reproduction
of the paper's filtered dataset. See protocol.json and the generated HTML report.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "trained_models/cf_feasibility_baseline"
UPSTREAM_COMMIT = "db715fa107cc1ca328ae6d7863de8a752ca13bdc"


@dataclass(frozen=True)
class Config:
    classifier_seed: int = 42
    classifier_epochs: int = 100
    classifier_lr: float = 0.01  # Repository value; paper text says 0.001.
    classifier_batch: int = 32
    epochs: int = 50
    batch: int = 2048
    lr: float = 0.01
    weight_decay: float = 0.01
    latent_dim: int = 10
    mc_samples: int = 50
    margin: float = 0.764
    validity_weight: float = 29.0
    age_weight: float = 192.0
    eval_samples: int = 10
    eval_batch: int = 512
    eval_seed: int = 20260927


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def load_data(out):
    a = np.load(out / "adult_inputs.npz")
    meta = json.loads((out / "data_manifest.json").read_text())
    assert hashlib.sha256(a["x"].tobytes()).hexdigest() == meta["encoded_x_sha256"]
    return a, meta


class BlackBox(nn.Module):
    """Exact repository classifier: two linear layers, without a ReLU."""
    def __init__(self, n_inputs=29):
        super().__init__()
        self.predict_net = nn.Sequential(nn.Linear(n_inputs, 10), nn.Linear(10, 2))

    def forward(self, x):
        return self.predict_net(x)


def mlp(widths, sigmoid=False):
    layers = []
    for start, end in zip(widths[:-2], widths[1:-1]):
        layers.extend([nn.Linear(start, end), nn.BatchNorm1d(end), nn.Dropout(0.1), nn.ReLU()])
    layers.append(nn.Linear(widths[-2], widths[-1]))
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class ModelApproxCF(nn.Module):
    """Upstream conditional VAE architecture, shared by both experiment arms."""
    def __init__(self, n_inputs=29, latent_dim=10):
        super().__init__()
        self.encoder_mean = mlp([n_inputs + 1, 20, 16, 14, 12, latent_dim])
        self.encoder_var = mlp([n_inputs + 1, 20, 16, 14, 12, latent_dim], sigmoid=True)
        self.decoder_mean = mlp([latent_dim + 1, 12, 14, 16, 20, n_inputs], sigmoid=True)

    def encode(self, x, target):
        xc = torch.cat([x, target.to(x.dtype).reshape(-1, 1)], dim=-1)
        return self.encoder_mean(xc), 0.5 + self.encoder_var(xc)

    def forward(self, x, target, samples=50):
        mean, variance = self.encode(x, target)
        target_column = target.to(x.dtype).reshape(-1, 1)
        # Keep separate decoder calls: upstream BatchNorm statistics are per draw.
        outputs = []
        for _ in range(samples):
            z = mean + variance.sqrt() * torch.randn_like(mean)
            outputs.append(self.decoder_mean(torch.cat([z, target_column], dim=-1)))
        return torch.stack(outputs), mean, variance


class AdultPolicy:
    """Feature blocks derived from saved metadata, never guessed column offsets."""
    fixed_names = ("race", "gender", "marital_status")
    banned_names = {"workclass": "Other/Unknown", "occupation": "Other/Unknown"}

    def __init__(self, metadata):
        self.features = metadata["features"]
        self.by_name = {f["name"]: f for f in self.features}
        self.blocks = [(f["start"], f["end"]) for f in self.features if f["categorical"]]
        self.numeric = [f for f in self.features if not f["categorical"]]
        self.banned = {name: self.by_name[name]["start"] + self.by_name[name]["categories"].index(cat)
                       for name, cat in self.banned_names.items()}
        self.age_index = self.by_name["age"]["start"]

    def project(self, raw, x, hard=False):
        """Copy fixed blocks, age floor, and allowed categorical simplex.

        Upstream decoder outputs sigmoid scores, not logits: normalize the allowed
        positive scores during training; argmax produces one-hot at evaluation.
        This imposes the same feasible set as CounterNet, with a suitable soft
        categorical operation for this decoder. x broadcasts over sample draws.
        """
        parts = []
        for f in self.features:
            s, e = f["start"], f["end"]
            block = raw[..., s:e]
            if f["name"] in self.fixed_names:
                block = torch.broadcast_to(x[..., s:e], block.shape)
            elif f["categorical"]:
                if f["name"] in self.banned:
                    allowed = torch.ones(e - s, dtype=raw.dtype, device=raw.device)
                    allowed[self.banned[f["name"]] - s] = 0
                    block = block * allowed
                if hard:
                    block = F.one_hot(block.argmax(-1), e - s).to(raw.dtype)
                else:
                    block = block / block.sum(-1, keepdim=True).clamp_min(1e-12)
            else:
                block = block.clamp(0, 1)
                if f["name"] == "age":
                    block = torch.maximum(block, x[..., s:e])
            parts.append(block)
        return torch.cat(parts, dim=-1)

    def decode(self, raw):
        """Common representation cleanup; adds NO age/immutability/ban policy."""
        parts = []
        for f in self.features:
            block = raw[..., f["start"]:f["end"]]
            if f["categorical"]:
                block = F.one_hot(block.argmax(-1), block.shape[-1]).to(raw.dtype)
            else:
                block = block.clamp(0, 1)
            parts.append(block)
        return torch.cat(parts, dim=-1)


def objective(raw, mean, variance, x, target, classifier, policy, cfg, extended):
    cf = policy.project(raw, x) if extended else raw
    # Preserve upstream's s:-1 slice, including its omission of the final gender
    # coordinate. Both arms use it; reported L1 metrics include ALL 29 coordinates.
    cat_start = len(policy.numeric)
    proximity = (cf[..., cat_start:-1] - x[..., cat_start:-1]).abs().sum(-1)
    for f in policy.numeric:
        j = f["start"]
        proximity = proximity + (f["max"] - f["min"]) * (cf[..., j] - x[..., j]).abs()
    simplex = sum((1 - cf[..., s:e].sum(-1)).abs() for s, e in policy.blocks)
    kl = 0.5 * (mean.square() + variance - variance.log() - 1).mean()
    logits = classifier(cf)
    validity = cf.new_zeros(())
    # Preserve upstream's sum of per-target-class means, not a class-frequency
    # weighted overall mean. Empty classes are skipped to avoid mean(empty)=NaN.
    for label in (0, 1):
        mask = target == label
        if mask.any():
            difference = logits[:, mask, label].sigmoid() - logits[:, mask, 1 - label].sigmoid()
            validity = validity + F.relu(cfg.margin - difference).mean()
    # Both arms retain the native penalty on the RAW age proposal. Computing it
    # after the extension's age floor would make the penalty identically zero.
    age_penalty = F.relu(x[..., policy.age_index] - raw[..., policy.age_index]).mean()
    total = (proximity + simplex).mean() + kl + cfg.validity_weight * validity + cfg.age_weight * age_penalty
    return total, dict(total=total, proximity=proximity.mean(), simplex=simplex.mean(),
                       kl=kl, validity_hinge=validity, age_penalty=age_penalty)


def setup(out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        config=asdict(cfg), upstream_commit=UPSTREAM_COMMIT,
        data="Saved ReLax Adult x/y and exact CounterNet train/test indices; no age/income filtering.",
        classifier="Repository BlackBox, 29→10→2, no activation; trained once, frozen for all CF runs.",
        target="1 - classifier(x).argmax(-1); both predicted directions, all training rows.",
        tuning="None. Fixed published Adult-Age margin/validity/feasibility weights; final epoch only.",
        original="Native conditional VAE and C1 age penalty, adapted to the common data protocol.",
        extended="Same VAE/loss plus fixed blocks, age floor and allowed-category normalization before CF prediction/proximity.",
        projection="Normalize allowed sigmoid scores in training; hard argmax at test; native age penalty remains on raw output.",
        evaluation="10 independent draws per test row, all counted equally, no best-of selection. Common hard categorical decoding.",
        diagnostics="Native soft-output validity and inference-only projection of original samples; no additional trained model.",
        compatibility_changes=["CPU/device-neutral tensors; no unused torchvision/TensorFlow dependency",
                               "Guard empty target-class slices in the hinge loss; same formula otherwise",
                               "Remove age/income filtering and use the saved CounterNet splits in both arms"],
        preserved_quirks=["Classifier has no activation, unlike the paper description; repository LR is 0.01",
                          "Native categorical reconstruction slice omits the final encoded coordinate",
                          "Continuous reconstruction is weighted by raw feature range, unlike reported encoded-space L1"],
        software=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
    )
    path = out / "protocol.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["config"] != protocol["config"]:
            raise ValueError("Existing output has different settings; select a new --out directory.")
    save_json(path, protocol)
    return protocol


def train_classifier(out, cfg):
    a, meta = load_data(out)
    path = out / "classifier.pt"
    if path.exists():
        print("Reusing frozen classifier", flush=True)
        return
    seed_all(cfg.classifier_seed)
    model = BlackBox()
    train = a["train_indices"]
    x, y = torch.from_numpy(a["x"][train]), torch.from_numpy(a["y"][train])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.classifier_lr)
    rng = np.random.default_rng(cfg.classifier_seed)
    start, history = time.perf_counter(), []
    for epoch in range(1, cfg.classifier_epochs + 1):
        model.train()
        permutation = rng.permutation(len(x))
        losses = []
        # Source divides into floor(N/32) near-equal batches.
        for indices in np.array_split(permutation, len(x) // cfg.classifier_batch):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite classifier loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, train_cross_entropy=float(np.mean(losses))))
        if epoch == 1 or epoch % 10 == 0:
            print(f"classifier epoch={epoch}/{cfg.classifier_epochs} loss={np.mean(losses):.5f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(a["x"])).argmax(-1).numpy()
    results = dict(seed=cfg.classifier_seed, epochs=cfg.classifier_epochs,
                   training_seconds=time.perf_counter() - start, state_sha256=state_hash(model), history=history)
    for split in ("train", "test"):
        idx = a[split + "_indices"]
        truth, pred = a["y"][idx], predicted[idx]
        results[split] = dict(n=len(idx), accuracy=float((truth == pred).mean()),
                             confusion=[[int(((truth == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)],
                             predicted_low=int((pred == 0).sum()), predicted_high=int((pred == 1).sum()))
    torch.save(dict(state_dict=model.state_dict(), config=asdict(cfg), data_sha256=meta["encoded_x_sha256"]), path)
    save_json(out / "classifier_results.json", results)
    print("Classifier complete:", json.dumps(results["test"]), flush=True)


def frozen_classifier(out, cfg):
    checkpoint = torch.load(out / "classifier.pt", map_location="cpu", weights_only=True)
    # Generator objectives may differ (C1/C2); classifier settings must match.
    for key in ("classifier_seed", "classifier_epochs", "classifier_lr", "classifier_batch"):
        assert checkpoint["config"][key] == getattr(cfg, key), key
    metadata = json.loads((out / "data_manifest.json").read_text())
    assert checkpoint["data_sha256"] == metadata["encoded_x_sha256"]
    model = BlackBox()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model


def train_generator(out, cfg, variant, seed, objective_fn=objective):
    directory = out / f"{variant}_seed{seed}"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "generator.pt"
    if checkpoint.exists():
        print(f"Reusing {variant} seed={seed}", flush=True)
        return
    a, metadata = load_data(out)
    classifier = frozen_classifier(out, cfg)
    before = state_hash(classifier)
    policy = AdultPolicy(metadata)
    x = torch.from_numpy(a["x"][a["train_indices"]])
    with torch.no_grad():
        targets = 1 - classifier(x).argmax(-1)
    seed_all(seed)
    model = ModelApproxCF(latent_dim=cfg.latent_dim)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    permutation_rng = np.random.default_rng(seed)
    history, started = [], time.perf_counter()
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        permutation = permutation_rng.permutation(len(x))
        terms = []
        for offset in range(0, len(x), cfg.batch):
            indices = permutation[offset:offset + cfg.batch]
            batch, target = x[indices], targets[indices]
            optimizer.zero_grad(set_to_none=True)
            raw, mean, variance = model(batch, target, cfg.mc_samples)
            loss, stats = objective_fn(raw, mean, variance, batch, target, classifier, policy, cfg, variant == "extended")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {variant} seed {seed} epoch {epoch}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite generator gradient")
            optimizer.step()
            terms.append({k: float(v.detach()) for k, v in stats.items()})
        row = dict(epoch=epoch, elapsed_seconds=time.perf_counter() - started,
                   **{k: float(np.mean([t[k] for t in terms])) for k in terms[0]})
        history.append(row)
        save_json(directory / "history.json", history)
        if epoch == 1 or epoch % 5 == 0:
            print(f"{variant} seed={seed} epoch={epoch}/{cfg.epochs} loss={row['total']:.4f} hinge={row['validity_hinge']:.4f} raw_age={row['age_penalty']:.5f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    assert before == state_hash(classifier), "Classifier changed during CF training"
    duration = time.perf_counter() - started
    torch.save(dict(state_dict=model.state_dict(), config=asdict(cfg), seed=seed, variant=variant,
                    classifier_sha256=before, initial_sha256=initial_hash,
                    data_sha256=metadata["encoded_x_sha256"]), checkpoint)
    save_json(directory / "training.json", dict(variant=variant, seed=seed, epochs=cfg.epochs,
               seconds=duration, initial_sha256=initial_hash, final_sha256=state_hash(model), classifier_sha256=before))


def candidate_metrics(cfs, x, predicted, cf_predicted, policy):
    """Per-draw/per-person outcomes, for common hard-decoded feature vectors."""
    result = dict(valid=(cf_predicted != predicted[None, :]),
                  proximity=np.abs(cfs - x[None, :, :]).sum(-1))
    for name in policy.fixed_names:
        f = policy.by_name[name]
        result["viol_" + name] = (np.abs(cfs[..., f["start"]:f["end"]] - x[None, :, f["start"]:f["end"]]).sum(-1) > 1e-6)
    j = policy.age_index
    result["viol_age_decrease"] = cfs[..., j] < x[None, :, j] - 1e-6
    for name, index in policy.banned.items():
        result["viol_" + name + "_unknown"] = cfs[..., index] > 0.5
    violations = [v for k, v in result.items() if k.startswith("viol_")]
    result["feasible"] = ~np.logical_or.reduce(violations)
    result["valid_feasible"] = result["valid"] & result["feasible"]
    return result


def summarize_candidates(metrics, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(n_people=0)
    m = {k: v[:, mask] for k, v in metrics.items()}
    summary = {k: float(v.mean()) for k, v in m.items()}
    summary.update(n_people=n, draws_per_person=int(m["valid"].shape[0]))
    for suffix, selected in [("valid", m["valid"]), ("valid_feasible", m["valid_feasible"])]:
        summary["proximity_" + suffix] = float(m["proximity"][selected].mean()) if selected.any() else None
    summary["validity_given_feasible"] = float(m["valid"][m["feasible"]].mean()) if m["feasible"].any() else None
    return summary


def evaluate_generator(out, cfg, variant, seed, metrics_fn=candidate_metrics):
    a, metadata = load_data(out)
    directory = out / f"{variant}_seed{seed}"
    classifier = frozen_classifier(out, cfg)
    checkpoint = torch.load(directory / "generator.pt", map_location="cpu", weights_only=True)
    assert checkpoint["config"] == asdict(cfg)
    assert checkpoint["classifier_sha256"] == state_hash(classifier)
    model = ModelApproxCF(latent_dim=cfg.latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    policy = AdultPolicy(metadata)
    x = torch.from_numpy(a["x"][a["test_indices"]])
    x_np = x.numpy()
    with torch.no_grad():
        predicted = classifier(x).argmax(-1).numpy()
    modes = ["decoded", "native_soft"] + (["inference_projection"] if variant == "original" else [])
    arrays = {mode: [] for mode in modes}
    predictions = {mode: [] for mode in modes}
    rng = torch.Generator().manual_seed(cfg.eval_seed + seed)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(x), cfg.eval_batch):
            batch = x[offset:offset + cfg.eval_batch]
            target = 1 - classifier(batch).argmax(-1)
            mean, variance = model.encode(batch, target)
            epsilon = torch.randn((cfg.eval_samples, len(batch), cfg.latent_dim), generator=rng)
            z = mean[None] + variance.sqrt()[None] * epsilon
            label = target.to(batch.dtype)[None, :, None].expand(cfg.eval_samples, -1, -1)
            raw = model.decoder_mean(torch.cat([z, label], -1).reshape(-1, cfg.latent_dim + 1)).reshape(cfg.eval_samples, len(batch), -1)
            soft = policy.project(raw, batch) if variant == "extended" else raw
            outputs = dict(decoded=policy.decode(soft), native_soft=soft)
            if variant == "original":
                outputs["inference_projection"] = policy.project(raw, batch, hard=True)
            for mode, values in outputs.items():
                arrays[mode].append(values.numpy())
                predictions[mode].append(classifier(values).argmax(-1).numpy())
    cohorts = dict(all=np.ones(len(x), dtype=bool), low_to_high=predicted == 0, high_to_low=predicted == 1)
    results = dict(variant=variant, seed=seed, classifier_sha256=state_hash(classifier),
                   evaluation_seconds=time.perf_counter() - start, cohorts={})
    saved = dict(test_indices=a["test_indices"], factual_prediction=predicted)
    for mode in modes:
        values = np.concatenate(arrays[mode], axis=1)
        preds = np.concatenate(predictions[mode], axis=1)
        saved["cf_" + mode] = values
        saved["prediction_" + mode] = preds
        if mode == "native_soft":
            results[mode] = {c: dict(valid=float((preds[:, mask] != predicted[None, mask]).mean()),
                                     proximity=float(np.abs(values[:, mask] - x_np[None, mask]).sum(-1).mean()))
                             for c, mask in cohorts.items() if mask.any()}
            continue
        metrics = metrics_fn(values, x_np, predicted, preds, policy)
        if variant == "extended" or mode == "inference_projection":
            assert metrics["feasible"].all(), "Projection violated encoded policy"
        results["cohorts"][mode] = {c: summarize_candidates(metrics, mask) for c, mask in cohorts.items()}
    np.savez_compressed(directory / "test_candidates.npz", **saved)
    save_json(directory / "evaluation.json", results)
    print(f"EVAL {variant} seed={seed}: " + json.dumps(results["cohorts"]["decoded"]["all"]), flush=True)
    return results


def aggregate(out):
    protocol = json.loads((out / "protocol.json").read_text())
    rows, runs = [], []
    for file in sorted(out.glob("*_seed*/evaluation.json")):
        run = json.loads(file.read_text())
        runs.append(run)
        for mode, cohorts in run["cohorts"].items():
            for cohort, metrics in cohorts.items():
                rows.append(dict(variant=run["variant"], seed=run["seed"], mode=mode, cohort=cohort, **metrics))
    if not rows:
        raise ValueError("No completed evaluations")
    fields = list(rows[0])
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    keys = sorted({(r["variant"], r["mode"], r["cohort"]) for r in rows})
    for variant, mode, cohort in keys:
        selected = [r for r in rows if (r["variant"], r["mode"], r["cohort"]) == (variant, mode, cohort)]
        values = {}
        for metric in [k for k in fields if k not in ("variant", "seed", "mode", "cohort")]:
            samples = [r[metric] for r in selected if r[metric] is not None]
            values[metric] = dict(mean=float(np.mean(samples)), sd=float(np.std(samples, ddof=1)) if len(samples) > 1 else None) if samples else None
        summary["/".join([variant, mode, cohort])] = dict(seeds=[r["seed"] for r in selected], metrics=values)
    paired = {}
    for cohort in ("all", "low_to_high", "high_to_low"):
        original = {r["seed"]: r for r in rows if r["variant"] == "original" and r["mode"] == "decoded" and r["cohort"] == cohort}
        extended = {r["seed"]: r for r in rows if r["variant"] == "extended" and r["mode"] == "decoded" and r["cohort"] == cohort}
        seeds = sorted(original.keys() & extended.keys())
        paired[cohort] = dict(seeds=seeds, metrics={})
        for metric in ("valid", "proximity", "feasible", "valid_feasible"):
            diffs = [extended[s][metric] - original[s][metric] for s in seeds]
            if diffs:
                paired[cohort]["metrics"][metric] = dict(differences=diffs, mean=float(np.mean(diffs)), sd=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None)
    save_json(out / "summary.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
              classifier=json.loads((out / "classifier_results.json").read_text()), summary=summary, paired=paired, runs=runs))
    print("Wrote results.csv and summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["classifier", "train", "evaluate", "aggregate", "all"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variant", choices=["original", "extended"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mc-samples", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(epochs=args.epochs, mc_samples=args.mc_samples)
    setup(args.out, cfg)
    if args.command in ("classifier", "all"):
        train_classifier(args.out, cfg)
    if args.command in ("train", "evaluate"):
        if args.command == "train":
            train_generator(args.out, cfg, args.variant, args.seed)
        evaluate_generator(args.out, cfg, args.variant, args.seed)
    if args.command == "all":
        for seed in args.seeds:
            for variant in ("original", "extended"):
                train_generator(args.out, cfg, variant, seed)
                evaluate_generator(args.out, cfg, variant, seed)
    if args.command in ("aggregate", "all"):
        aggregate(args.out)


if __name__ == "__main__":
    main()
