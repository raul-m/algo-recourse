"""Thesis experiment: enforce immutability of race, gender, marital_status in CounterNet CEs.

Mechanisms compared on the Adult dataset (evaluation on the held-out test split only):

  1. projection           — the features are marked immutable in the DataModule, so
                            `apply_constraints` copies them from the input x in every
                            forward pass (train: soft mode, test: hard mode). This is
                            the mechanism of the paper's feasibility experiment (§4.2),
                            extended from {race, gender} to also cover marital_status.

  2. actionability (λ4)   — the features are NOT projected (for none of the three).
                            A fourth loss term L4 = masked mean of ½(x - x')² over the
                            immutable columns is added to the CF generator objective
                            (stage 2), weighted by λ4. Swept over λ4 ∈ {0, 1, 5, 20}
                            by default; λ4 = 0 is the paper's base CounterNet
                            (no enforcement baseline).

  5. invariant predictor  — no projection, no L4. A fifth loss term L5 regularizes
     (λ5, A/B)              the *predictor* (stage 1) to be insensitive to the
                            immutable features, removing the generator's validity
                            incentive to change them. Two versions, swept over λ5:
                            A 'consistency' — penalize (ŷ(x) − ŷ(x̃))² with x̃ the
                            batch-shuffled immutable block (real profiles, joint
                            distribution preserved); B 'gradient' — penalize the
                            input-gradient on the immutable columns (local only).

  6. combined (λ4 + λ5)   — L5 on stage 1 (A, λ5 = 10) and L4 on stage 2, swept
                            over λ4. Tests whether generator pressure works better
                            once the predictor's incentive is gone. (Empirically:
                            the opposite — the invariant encoder strips demographic
                            information from the shared latent, so the generator
                            can no longer copy per-instance demographics and L4's
                            partial effect is erased.)

  7. reachability         — extends variant 1's hard projection with an Adult
     projection             reachability rule: race, gender, and marital_status are
                            copied exactly, while age may stay the same or increase
                            but may never decrease. Mutable workclass and occupation
                            may not select their explicit 'Other/Unknown' category.
                            All categorical outputs remain known one-hot categories;
                            immutable race='Other' remains legitimate and is preserved.

  8. recourse-aware       — variant 1's projection plus a stage-1 regularizer
     predictor (λR)         R_recourse (working note §8): mean over predicted-
                            unfavorable instances of a margin loss on the
                            predictor's output at the feasibility-projected CF
                            candidate (stop-gradient: the candidate is fixed,
                            the generator gets no stage-1 gradient). Trades
                            prediction loss against the existence/margin of
                            feasible recourse. Smoothed-hinge ℓ_target swept
                            over λR; quadratic ablation at λR = 1.

  AR. actionability      — variant 7's reachability projection in the CF network,
      recourse (λ4)        plus L3's proximity form added to the *predictor*
                           objective as L4 (stage 1), gradients flowing through
                           the projected candidate into Encoder/Predictor with
                           the CF generator stop-gradiented. Swept over λ4, and
                           additionally under two schedules that charge L4 less
                           often: 'every_k' (only on steps where step % k == 0)
                           and 'after_k' (only once step >= k, i.e. applied to
                           an already-converged model). The step counter lives
                           in the optimizer-state tuple because the trainer jits
                           training_step. A two-phase variant trains jointly with
                           no L4 first, then fine-tunes only the predictor on
                           λ1·L1 + λ4·L4 with the CF generator frozen and stage 2
                           switched off. Reported on actionability_recourse.html.

The actionability module is generic: the penalized columns are derived from feature
names (`penalized_features` in its config) and the dataset's own one-hot layout — it
is not tied to this dataset, feature set, or encoding width. Per feature, penalty and
projection are mutually exclusive (a projected feature would make L4 identically zero,
so overlapping the two is rejected); mixed setups project some features and penalize
others, on disjoint sets.

Metrics per variant: accuracy, validity, proximity (raw L1, /d, and over the mutable
columns only), per-feature violation rates, age-direction violations, and categorical
one-hot validity. Trained models are cached in
trained_models/immutability_experiment/<variant>/params.pkl and reused if present.
"""
import argparse
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
import pandas as pd
from pydantic import Field

import relax
from relax.methods import CounterNet
from relax.methods.counternet import (CounterNetConfig, CounterNetTrainingModule,
                                      partition_trainable_params)
from relax.legacy.utils import grad_update
from relax.evaluate import compute_validity, compute_proximity, compute_sparsity

IMMUTABLE = ["race", "gender", "marital_status"]
AGE = "age"
DISALLOWED_MUTABLE_CATEGORIES = {
    "workclass": ["Other/Unknown"],
    "occupation": ["Other/Unknown"],
}
EPOCHS = 100
FINETUNE_EPOCHS = 20
BATCH_SIZE = 128
TRAIN_SEED = 42
OUT_DIR = Path("trained_models/immutability_experiment")


def load_dm(immutable_names: list[str]):
    """Load the Adult data module with exactly `immutable_names` marked immutable."""
    dm = relax.load_data("adult")
    for feat in dm.features:
        feat._is_immutable = feat.name in immutable_names
    return dm


def feature_col_mask(dm, names: list[str]) -> jnp.ndarray:
    """0/1 mask over the input columns, 1 = column belongs to a feature in `names`.

    Built with numpy (not jnp) so it is always a concrete array, even if first
    accessed inside a jit trace.
    """
    mask = np.zeros(dm.xs.shape[1], dtype=np.float32)
    for feat, (s, e) in zip(dm.features, dm.features.feature_indices):
        if feat.name in names:
            mask[s:e] = 1.0
    return jnp.asarray(mask)


class ReachabilityProjectionConfig(CounterNetConfig):
    """Feature-level policy for hard feasibility/reachability projection.

    ``fixed_features`` are copied from the original input. Features in
    ``nondecreasing_features`` are lower-bounded by their original normalized value.
    The policy is expressed in feature names and resolved through the data module's
    layout, so it is reusable across transformed column positions and widths.
    """

    fixed_features: List[str] = Field(
        [], description="Feature names copied exactly from the original input."
    )
    nondecreasing_features: List[str] = Field(
        [], description="Continuous features that may stay fixed or increase only."
    )
    disallowed_categories: Dict[str, List[str]] = Field(
        {}, description="Known category labels that mutable generated features may "
                        "not select. Fixed-feature copying takes precedence."
    )
    nondecreasing_ordinal_features: List[str] = Field(
        [], description="Categorical features whose one-hot categories are stored in "
                        "ascending order and may only move up (e.g. OULAD's age_band). "
                        "Enforced by masking, per row, the logits of every category "
                        "below the factual's own category."
    )


class ReachabilityProjectionModule(CounterNetTrainingModule):
    """CounterNet with a feature-level reachability projection.

    The data module first applies its ordinary domain constraints (known-category
    softmax/argmax and continuous clipping). This module then copies fixed features
    and applies monotone lower bounds to configured continuous features. The same
    projection is used in training and held-out generation, just like CounterNet's
    built-in immutability projection.
    """

    def __init__(self, config: ReachabilityProjectionConfig | dict):
        config = (config if isinstance(config, ReachabilityProjectionConfig)
                  else ReachabilityProjectionConfig(**config))
        super().__init__(config)
        self._fixed_mask = None
        self._nondecreasing_mask = None
        self._disallowed_category_mask = None
        self._ordinal_blocks = None

    def init_net_opt(self, data_module, key):
        out = super().init_net_opt(data_module, key)
        self._validate_policy()
        _ = (self.fixed_mask, self.nondecreasing_mask, self.disallowed_category_mask,
             self.ordinal_blocks)
        return out

    def _validate_policy(self):
        dm = self._data_module
        configured = (set(self.configs.fixed_features)
                      | set(self.configs.nondecreasing_features)
                      | set(self.configs.nondecreasing_ordinal_features)
                      | set(self.configs.disallowed_categories))
        by_name = {feat.name: feat for feat in dm.features}
        unknown = configured - set(by_name)
        if unknown:
            raise ValueError(f"Unknown reachability-policy features: {sorted(unknown)}")
        overlap = (set(self.configs.fixed_features)
                   & set(self.configs.nondecreasing_features))
        if overlap:
            raise ValueError(
                "A feature cannot be both fixed and non-decreasing: "
                f"{sorted(overlap)}"
            )
        categorical_monotone = [
            name for name in self.configs.nondecreasing_features
            if by_name[name].is_categorical
        ]
        if categorical_monotone:
            raise ValueError(
                "Non-decreasing projection requires continuous/ordinal scalar features, "
                f"not categorical one-hot groups: {sorted(categorical_monotone)}"
            )
        ordinal_overlap = (set(self.configs.nondecreasing_ordinal_features)
                           & (set(self.configs.fixed_features)
                              | set(self.configs.nondecreasing_features)))
        if ordinal_overlap:
            raise ValueError(
                "An ordinal non-decreasing feature cannot also be fixed or a "
                f"continuous non-decreasing feature: {sorted(ordinal_overlap)}"
            )
        non_categorical_ordinal = [
            name for name in self.configs.nondecreasing_ordinal_features
            if not by_name[name].is_categorical
        ]
        if non_categorical_ordinal:
            raise ValueError(
                "Ordinal non-decreasing projection requires categorical one-hot "
                f"features; use nondecreasing_features for scalars: "
                f"{sorted(non_categorical_ordinal)}"
            )
        fixed_with_bans = (set(self.configs.fixed_features)
                           & set(self.configs.disallowed_categories))
        if fixed_with_bans:
            raise ValueError(
                "Disallowed generated categories may only be configured for mutable "
                "features; fixed values must always be preserved exactly: "
                f"{sorted(fixed_with_bans)}"
            )
        for name, labels in self.configs.disallowed_categories.items():
            feat = by_name[name]
            if not feat.is_categorical:
                raise ValueError(
                    f"Disallowed categories require a categorical feature, got {name!r}."
                )
            # labels arrive as strings (the config's declared type); compare on the
            # string form so integer-coded categories (e.g. credit's EDUCATION) match
            categories = [str(c) for c in
                          np.asarray(feat.transformation.transformer.categories_).ravel()]
            unknown_labels = set(map(str, labels)) - set(categories)
            if unknown_labels:
                raise ValueError(
                    f"Unknown category labels for {name!r}: {sorted(unknown_labels)}; "
                    f"known labels are {categories}"
                )
            if len(set(labels)) >= len(categories):
                raise ValueError(
                    f"At least one generated category must remain allowed for {name!r}."
                )

    @property
    def fixed_mask(self) -> jnp.ndarray:
        if self._fixed_mask is None:
            self._validate_policy()
            self._fixed_mask = feature_col_mask(
                self._data_module, self.configs.fixed_features)
        return self._fixed_mask

    @property
    def nondecreasing_mask(self) -> jnp.ndarray:
        if self._nondecreasing_mask is None:
            self._validate_policy()
            self._nondecreasing_mask = feature_col_mask(
                self._data_module, self.configs.nondecreasing_features)
        return self._nondecreasing_mask

    @property
    def disallowed_category_mask(self) -> jnp.ndarray:
        if self._disallowed_category_mask is None:
            self._validate_policy()
            mask = np.zeros(self._data_module.xs.shape[1], dtype=bool)
            for feat, (start, end) in zip(
                    self._data_module.features,
                    self._data_module.features.feature_indices):
                labels = self.configs.disallowed_categories.get(feat.name, [])
                if labels:
                    categories = [str(c) for c in np.asarray(
                        feat.transformation.transformer.categories_).ravel()]
                    for label in labels:
                        mask[start + categories.index(str(label))] = True
                    assert end - start == len(categories)
            self._disallowed_category_mask = jnp.asarray(mask)
        return self._disallowed_category_mask

    @property
    def ordinal_blocks(self) -> tuple:
        """(start, end) column spans of the ascending-ordinal one-hot features."""
        if self._ordinal_blocks is None:
            self._validate_policy()
            names = set(self.configs.nondecreasing_ordinal_features)
            self._ordinal_blocks = tuple(
                (int(s), int(e)) for feat, (s, e) in zip(
                    self._data_module.features,
                    self._data_module.features.feature_indices)
                if feat.name in names)
        return self._ordinal_blocks

    def apply_reachability_projection(self, x, cf, hard: bool):
        """Apply domain constraints, exact fixes, then monotone lower bounds."""
        # Mask logits before the data module's softmax/argmax. This renormalizes over
        # allowed known categories in training and selects only an allowed one-hot
        # category at inference. Fixed features are deliberately not eligible here.
        neg_inf = jnp.asarray(jnp.finfo(cf.dtype).min, dtype=cf.dtype)
        cf = jnp.where(self.disallowed_category_mask, neg_inf, cf)
        # Per-row monotone floor on ascending-ordinal one-hot blocks: every category
        # ranked below the factual's own is made unreachable, so the data module's
        # softmax/argmax can only stay put or move up.
        for start, end in self.ordinal_blocks:
            rank_x = jnp.argmax(x[:, start:end], axis=1)
            below = jnp.arange(end - start)[None, :] < rank_x[:, None]
            cf = cf.at[:, start:end].set(
                jnp.where(below, neg_inf, cf[:, start:end]))
        cf = self._data_module.apply_constraints(x, cf, hard=hard)
        cf = jnp.where(self.fixed_mask.astype(bool), x, cf)
        cf = jnp.where(
            self.nondecreasing_mask.astype(bool), jnp.maximum(cf, x), cf)
        return cf

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def forward(self, params, rng_key, x, is_training: bool = True):
        y_pred, cf = self.net.apply(params, rng_key, x, is_training=is_training)
        cf = self.apply_reachability_projection(x, cf, hard=not is_training)
        cf_y, _ = self.net.apply(params, rng_key, cf, is_training=is_training)
        return y_pred, cf, cf_y

    @partial(jax.jit, static_argnames=["self"])
    def generate_cf(self, params, rng_key, x):
        _, cf = self.net.apply(params, rng_key, x, is_training=False)
        return self.apply_reachability_projection(x, cf, hard=True)


class PredictorProximityConfig(ReachabilityProjectionConfig):
    """Reachability-projection config plus a stage-1 proximity weight.

    `lambda_4` weights the actionability-recourse loss L4 — the same functional
    form as the stage-2 proximity L3, evaluated on the reachability-projected CF
    but added to the *predictor's* objective (stage 1).
    """

    lambda_4: float = Field(
        1.0, description="Weight of the stage-1 proximity loss L4 on the predictor."
    )


class PredictorProximityReachabilityModule(ReachabilityProjectionModule):
    """Actionability-recourse experiment: v7's reachability projection in the CF
    network, plus the L3 proximity form added to the predictor as L4 (stage 1).

        stage 1:  λ1·L1 + λ4·L4,   L4 = ½·mean (x − x″)²   (≡ L3's form)
        stage 2:  λ2·L2 + λ3·L3                             (unchanged)

    L4 contains the predictor's parameters only *through* the candidate
    x″ = reachability_projection(G(z ⊕ p)): its stage-1 gradient flows through
    z and p into the Encoder and Predictor, while the Explainer's own weights
    are stop-gradiented — they remain stage-2 property, so the generator is
    still trained exactly once per batch. The mechanism is therefore: shape the
    shared representation (and the p-features the generator reads) so that the
    frozen generator's feasible CF lands closer to the input — a predictor
    co-optimized for cheap actionable recourse.
    """

    def __init__(self, config: PredictorProximityConfig | dict):
        config = (config if isinstance(config, PredictorProximityConfig)
                  else PredictorProximityConfig(**config))
        super().__init__(config)

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)

        # freeze the generator's weights for stage 1: L4's gradient reaches the
        # Encoder/Predictor through the candidate x″ only, never the Explainer
        explainer_params, backbone_params = partition_trainable_params(
            params, trainable_name="counter_net_model/Explainer")
        params_gen_frozen = hk.data_structures.merge(
            jax.lax.stop_gradient(explainer_params), backbone_params)
        _, cf = self.net.apply(params_gen_frozen, rng_key, x,
                               is_training=is_training)
        cf = self.apply_reachability_projection(x, cf, hard=not is_training)
        loss_4 = self.loss_fn_3(x, cf)  # identical functional form to L3
        return self.configs.lambda_1 * loss_1 + self.configs.lambda_4 * loss_4


class ImmutableGradientConfig(ReachabilityProjectionConfig):
    """Reachability-projection config plus the immutable input-gradient penalty."""

    invariant_features: List[str] = Field(
        [], description="Features whose input-gradient the predictor is penalized on."
    )
    lambda_4: float = Field(
        1.0, description="Weight of the stage-1 gradient penalty L4."
    )


class ImmutableGradientPredictorModule(ReachabilityProjectionModule):
    """Experiment 5: L4 = masked input-gradient penalty on the predictor.

        stage 1:  λ1·L1 + λ4·L4,   L4 = Σ (m ⊙ ∂ŷ/∂x)² / (N·|m|)
        stage 2:  λ2·L2 + λ3·L3                          (unchanged)

    Where E1–E4.1 charged the predictor for the *cost* of the counterfactual, this
    charges it for *local sensitivity* to the immutable coordinates: around each
    observed x, a small move along race/gender/marital_status should barely move the
    predicted probability. The CF network is unchanged — the reachability projection
    still copies those columns — so the guarantees are independent of this term.

    Same functional form as the immutability study's variant 5B, but on the
    reachability-projection base rather than an unconstrained generator.
    """

    def __init__(self, config: ImmutableGradientConfig | dict):
        config = (config if isinstance(config, ImmutableGradientConfig)
                  else ImmutableGradientConfig(**config))
        super().__init__(config)
        self._invariance_mask = None

    def init_net_opt(self, data_module, key):
        out = super().init_net_opt(data_module, key)
        _ = self.invariance_mask
        return out

    @property
    def invariance_mask(self) -> jnp.ndarray:
        if self._invariance_mask is None:
            names = self.configs.invariant_features
            if not names:
                raise ValueError("`invariant_features` must name at least one feature.")
            unknown = set(names) - {f.name for f in self._data_module.features}
            if unknown:
                raise ValueError(f"Unknown feature names: {sorted(unknown)}")
            self._invariance_mask = feature_col_mask(self._data_module, names)
        return self._invariance_mask

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)
        # rows are independent (no batch norm), so the gradient of the summed output
        # w.r.t. the batch input stacks the per-example input gradients
        grads = jax.grad(
            lambda xin: self.net.apply(
                params, rng_key, xin, is_training=is_training)[0].sum()
        )(x)
        loss_4 = ((grads * self.invariance_mask) ** 2).sum() / (
            x.shape[0] * self.invariance_mask.sum())
        return self.configs.lambda_1 * loss_1 + self.configs.lambda_4 * loss_4


class ScheduledPredictorProximityConfig(PredictorProximityConfig):
    """Predictor-proximity config with a schedule controlling when L4 is applied."""

    l4_schedule: str = Field(
        "always",
        description="'always' — L4 on every predictor update; 'every_k' — only on "
                    "steps where step %% k == 0 (intermittent pressure); 'after_k' — "
                    "only once step >= k (delayed pressure on a converged model)."
    )
    l4_k: int = Field(
        1, description="Period (every_k) or warm-up length (after_k), in training steps."
    )


class ScheduledPredictorProximityModule(PredictorProximityReachabilityModule):
    """L4 applied on a schedule rather than at every predictor update.

    Both schedules target the accumulation mechanism: L4's benefit is re-equilibrated
    by the next stage-2 update, while its distortion of the shared representation
    accumulates across steps. 'every_k' lowers the accumulation rate (stage 2 gets
    k-1 clean steps to recover between charges); 'after_k' applies the charge only to
    an already-converged representation.

    The step counter is carried inside the optimizer-state tuple (o1, o2, step) rather
    than as a Python attribute: the trainer wraps `training_step` in `jax.jit`, so a
    host-side counter would be incremented once at trace time and then frozen.
    """

    def __init__(self, config: ScheduledPredictorProximityConfig | dict):
        config = (config if isinstance(config, ScheduledPredictorProximityConfig)
                  else ScheduledPredictorProximityConfig(**config))
        if config.l4_schedule not in ("always", "every_k", "after_k"):
            raise ValueError(
                f"l4_schedule must be 'always', 'every_k' or 'after_k', "
                f"got {config.l4_schedule!r}")
        if config.l4_schedule != "always" and config.l4_k < 1:
            raise ValueError("l4_k must be >= 1 for a scheduled L4.")
        super().__init__(config)

    def init_net_opt(self, data_module, key):
        params, (opt_1_state, opt_2_state) = super().init_net_opt(data_module, key)
        return params, (opt_1_state, opt_2_state, jnp.asarray(0, dtype=jnp.int32))

    def _l4_gate(self, step):
        """1.0 on steps where L4 is charged, 0.0 otherwise (traced)."""
        kind, k = self.configs.l4_schedule, self.configs.l4_k
        if kind == "always":
            return jnp.asarray(1.0)
        if kind == "every_k":
            return (jnp.mod(step, k) == 0).astype(jnp.float32)
        return (step >= k).astype(jnp.float32)

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, l4_gate, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)

        explainer_params, backbone_params = partition_trainable_params(
            params, trainable_name="counter_net_model/Explainer")
        params_gen_frozen = hk.data_structures.merge(
            jax.lax.stop_gradient(explainer_params), backbone_params)
        _, cf = self.net.apply(params_gen_frozen, rng_key, x, is_training=is_training)
        cf = self.apply_reachability_projection(x, cf, hard=not is_training)
        loss_4 = self.loss_fn_3(x, cf)
        # gate multiplies the term, so on a skipped step the total stage-1 gradient
        # is exactly the L1 gradient — equivalent to L4 not being present at all
        return (self.configs.lambda_1 * loss_1
                + l4_gate * self.configs.lambda_4 * loss_4)

    @partial(jax.jit, static_argnames=["self"])
    def _predictor_step(self, params, opt_state, rng_key, batch, l4_gate):
        grads = jax.grad(self.pred_loss_fn)(params, rng_key, batch, l4_gate)
        return grad_update(grads, params, opt_state, self.opt_1)

    def training_step(self, params, opts_state, rng_key, batch):
        opt_1_state, opt_2_state, step = opts_state
        params, opt_1_state = self._predictor_step(
            params, opt_1_state, rng_key, batch, self._l4_gate(step))
        params, opt_2_state = self._explainer_step(
            params, opt_2_state, rng_key, batch)
        logs = self._training_step_logs(params, rng_key, batch)
        return logs, (params, (opt_1_state, opt_2_state, step + 1))


def finetune_predictor_on_l4(cfnet: CounterNet, dm, lambda_4: float, epochs: int,
                             seed: int = TRAIN_SEED, every_k: int = 1,
                             after_k: int = 0, l4_kind: str = "proximity",
                             imm_features: List[str] | None = None) -> CounterNet:
    """Phase 2 of the two-phase experiment: fine-tune the predictor on L4 alone.

    Phase 1 is ordinary joint CounterNet training (both stages, no L4). Here the CF
    generator is frozen *completely* — it receives no gradient and stage 2 is not run
    at all — and only the Encoder/Predictor/Head are updated, minimizing

        λ1·L1 + gate·λ4·L4,   L4 = ½·mean (x − x″)²,  x″ = P(G(z(θ) ⊕ p(θ)))

    `every_k` gates L4 inside phase 2 exactly as the intermittent schedule does during
    joint training: gate = 1 when step % every_k == 0, else 0, so on a skipped step the
    update is the plain L1 gradient. every_k = 1 charges on every step.

    This is the cleanest isolation of the stage-1 channel: with the generator frozen
    and stage 2 switched off, nothing re-equilibrates what the predictor does, so any
    reduction in recourse cost that the representation *can* produce should survive.
    Note that validity is absent from the objective — whether the CFs still cross the
    boundary is the measured outcome, not something the fine-tuning protects.
    """
    module = cfnet.module
    params = cfnet.params
    # build the projection masks eagerly: they are cached on the module, and a
    # first build inside the jit trace below would cache a leaked tracer
    _ = module.fixed_mask, module.nondecreasing_mask, module.disallowed_category_mask
    opt = optax.adam(learning_rate=module.configs.lr)
    backbone, generator = hk.data_structures.partition(
        lambda m, n, v: "counter_net_model/Explainer" not in m, params)
    opt_state = opt.init(backbone)

    imm_mask = feature_col_mask(dm, imm_features or IMMUTABLE)

    def loss_fn(backbone_params, generator_params, rng_key, batch, l4_gate):
        x, y = batch
        merged = hk.data_structures.merge(backbone_params, generator_params)
        y_pred, _ = module.net.apply(merged, rng_key, x, is_training=True)
        loss_1 = module.loss_fn_1(y_pred, y)
        if l4_kind == "gradient":
            # local insensitivity to the immutable coordinates; no CF involved
            g = jax.grad(lambda xin: module.net.apply(
                merged, rng_key, xin, is_training=True)[0].sum())(x)
            loss_4 = ((g * imm_mask) ** 2).sum() / (x.shape[0] * imm_mask.sum())
        else:
            # generator frozen: its params enter the graph only as constants
            frozen = hk.data_structures.merge(
                backbone_params, jax.lax.stop_gradient(generator_params))
            _, cf = module.net.apply(frozen, rng_key, x, is_training=True)
            cf = module.apply_reachability_projection(x, cf, hard=False)
            loss_4 = module.loss_fn_3(x, cf)
        return module.configs.lambda_1 * loss_1 + l4_gate * lambda_4 * loss_4

    @jax.jit
    def step(backbone_params, generator_params, opt_state, rng_key, batch, l4_gate):
        grads = jax.grad(loss_fn)(
            backbone_params, generator_params, rng_key, batch, l4_gate)
        upt, opt_state = grad_update(grads, backbone_params, opt_state, opt)
        return upt, opt_state

    xs, ys = dm["train"]
    n = xs.shape[0]
    key = jrand.PRNGKey(seed)
    step_i = 0
    for _ in range(epochs):
        key, shuffle_key = jrand.split(key)
        order = jrand.permutation(shuffle_key, n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            key, step_key = jrand.split(key)
            gate = jnp.asarray(
                1.0 if (step_i % every_k == 0 and step_i >= after_k) else 0.0)
            backbone, opt_state = step(
                backbone, generator, opt_state, step_key, (xs[idx], ys[idx]), gate)
            step_i += 1

    cfnet.params = hk.data_structures.merge(backbone, generator)
    return cfnet


class ScheduledImmutableGradientConfig(ImmutableGradientConfig):
    """Gradient-penalty config with a schedule controlling when L4 is charged."""

    l4_schedule: str = Field("always", description="'always' | 'every_k' | 'after_k'.")
    l4_k: int = Field(1, description="Period (every_k) or warm-up length (after_k), in steps.")


class ScheduledImmutableGradientModule(ImmutableGradientPredictorModule):
    """E5 under the same schedules as E2/E3: the gradient penalty is charged only on
    every k-th step, or only after a warm-up. Step counter lives in the optimizer
    state because the trainer jits `training_step`."""

    def __init__(self, config: ScheduledImmutableGradientConfig | dict):
        config = (config if isinstance(config, ScheduledImmutableGradientConfig)
                  else ScheduledImmutableGradientConfig(**config))
        if config.l4_schedule not in ("always", "every_k", "after_k"):
            raise ValueError(f"bad l4_schedule: {config.l4_schedule!r}")
        super().__init__(config)

    def init_net_opt(self, data_module, key):
        params, (o1, o2) = super().init_net_opt(data_module, key)
        return params, (o1, o2, jnp.asarray(0, dtype=jnp.int32))

    def _l4_gate(self, step):
        kind, k = self.configs.l4_schedule, self.configs.l4_k
        if kind == "always":
            return jnp.asarray(1.0)
        if kind == "every_k":
            return (jnp.mod(step, k) == 0).astype(jnp.float32)
        return (step >= k).astype(jnp.float32)

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, l4_gate, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)
        grads = jax.grad(
            lambda xin: self.net.apply(
                params, rng_key, xin, is_training=is_training)[0].sum())(x)
        loss_4 = ((grads * self.invariance_mask) ** 2).sum() / (
            x.shape[0] * self.invariance_mask.sum())
        return self.configs.lambda_1 * loss_1 + l4_gate * self.configs.lambda_4 * loss_4

    @partial(jax.jit, static_argnames=["self"])
    def _predictor_step(self, params, opt_state, rng_key, batch, l4_gate):
        grads = jax.grad(self.pred_loss_fn)(params, rng_key, batch, l4_gate)
        return grad_update(grads, params, opt_state, self.opt_1)

    def training_step(self, params, opts_state, rng_key, batch):
        o1, o2, step = opts_state
        params, o1 = self._predictor_step(params, o1, rng_key, batch, self._l4_gate(step))
        params, o2 = self._explainer_step(params, o2, rng_key, batch)
        logs = self._training_step_logs(params, rng_key, batch)
        return logs, (params, (o1, o2, step + 1))


class ActionabilityConfig(CounterNetConfig):
    """CounterNet config extended with the actionability loss settings."""

    penalized_features: List[str] = Field(
        [], description="Feature names penalized by L4. Must not overlap with features "
                        "flagged `is_immutable` in the data module (those are already "
                        "enforced by the projection, making L4 identically zero for them)."
    )
    lambda_4: float = Field(
        1.0, description="Weight of the actionability loss L4."
    )


class ActionabilityCounterNetModule(CounterNetTrainingModule):
    """CounterNet training module with an additional actionability loss L4.

    L4 penalizes changes on the columns of the immutable features:
        L4 = sum_over(batch, immutable cols) ½(x - x')² / (batch_size * n_immutable_cols)
    added to the stage-2 (CF generator) objective with weight lambda_4.

    The column mask is dataset-agnostic: it is derived from the hooked data module
    using the feature names in `config.penalized_features`. Naming a feature that is
    also flagged `is_immutable` (i.e. projected) is rejected: the projection copies
    those columns before L4 sees them, making the penalty identically zero.
    """

    def __init__(self, config: ActionabilityConfig | dict):
        config = config if isinstance(config, ActionabilityConfig) else ActionabilityConfig(**config)
        super().__init__(config)
        self._penalty_mask = None

    def init_net_opt(self, data_module, key):
        out = super().init_net_opt(data_module, key)
        _ = self.penalty_mask  # build eagerly, outside any jit trace
        return out

    @property
    def penalty_mask(self) -> jnp.ndarray:
        if self._penalty_mask is None:
            dm = self._data_module
            names = self.configs.penalized_features
            if not names:
                raise ValueError(
                    "`penalized_features` must name at least one feature."
                )
            projected = {f.name for f in dm.features if f.is_immutable} & set(names)
            if projected:
                raise ValueError(
                    f"{sorted(projected)} are flagged is_immutable in the data module, "
                    "so the projection already copies them and L4 would be identically "
                    "zero for these features. Penalize a feature or project it — not both."
                )
            unknown = set(names) - {f.name for f in dm.features}
            if unknown:
                raise ValueError(f"Unknown feature names: {sorted(unknown)}")
            self._penalty_mask = feature_col_mask(dm, names)
        return self._penalty_mask

    @partial(jax.jit, static_argnames=["self"])
    def loss_fn_4(self, x, cf):
        elementwise = optax.l2_loss(cf, x) * self.penalty_mask
        return elementwise.sum() / (x.shape[0] * self.penalty_mask.sum())

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def exp_loss_fn(
        self,
        trainable_params,
        non_trainable_params,
        rng_key,
        batch,
        is_training: bool = True,
    ):
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        x, y = batch
        y_pred, cf, cf_y = self.forward(params, rng_key, x, is_training=is_training)
        y_prime = 1 - jnp.round(y_pred)
        loss_2, loss_3 = self.loss_fn_2(cf_y, y_prime), self.loss_fn_3(x, cf)
        loss_4 = self.loss_fn_4(x, cf)
        return (
            self.configs.lambda_2 * loss_2
            + self.configs.lambda_3 * loss_3
            + self.configs.lambda_4 * loss_4
        )


class InvarianceConfig(CounterNetConfig):
    """CounterNet config extended with the predictor-invariance regularizer L5."""

    invariant_features: List[str] = Field(
        [], description="Feature names the predictor is regularized to ignore."
    )
    lambda_5: float = Field(
        1.0, description="Weight of the invariance loss L5."
    )
    invariance_kind: str = Field(
        "consistency",
        description="'consistency' (option A: penalize the prediction gap under "
                    "batch-shuffled immutable blocks) or 'gradient' (option B: "
                    "penalize the input-gradient on the immutable columns)."
    )


class InvariantPredictorCounterNetModule(CounterNetTrainingModule):
    """CounterNet training module with a stage-1 invariance regularizer L5.

    Unlike L4 (which fights the CF generator), L5 acts on the *predictor*: if the
    prediction cannot be moved by the immutable features, the generator has no
    validity incentive to change them. The stage-2 objective is left untouched.

    Option A ('consistency'):
        L5 = mean ½(ŷ(x) − ŷ(x̃))², where x̃ carries the immutable-feature block of
        another instance in the batch — one shared row permutation for all masked
        columns, so every x̃ is a real, empirically occurring demographic profile
        and the joint distribution of the immutable features is preserved.

    Option B ('gradient'):
        L5 = Σ (m ⊙ ∂ŷ/∂x)² / (N·|m|) — first-order insensitivity to the immutable
        columns at the data points (local flatness; category flips are finite jumps,
        so this is the weaker condition of the two).
    """

    def __init__(self, config: InvarianceConfig | dict):
        config = config if isinstance(config, InvarianceConfig) else InvarianceConfig(**config)
        if config.invariance_kind not in ("consistency", "gradient"):
            raise ValueError(
                f"invariance_kind must be 'consistency' or 'gradient', got {config.invariance_kind!r}"
            )
        super().__init__(config)
        self._invariance_mask = None

    def init_net_opt(self, data_module, key):
        out = super().init_net_opt(data_module, key)
        _ = self.invariance_mask  # build eagerly, outside any jit trace
        return out

    @property
    def invariance_mask(self) -> jnp.ndarray:
        if self._invariance_mask is None:
            dm = self._data_module
            names = self.configs.invariant_features
            if not names:
                raise ValueError("`invariant_features` must name at least one feature.")
            unknown = set(names) - {f.name for f in dm.features}
            if unknown:
                raise ValueError(f"Unknown feature names: {sorted(unknown)}")
            self._invariance_mask = feature_col_mask(dm, names)
        return self._invariance_mask

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        apply_key, perm_key = jrand.split(rng_key)
        y_pred, _ = self.net.apply(params, apply_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)
        if self.configs.invariance_kind == "consistency":
            # same apply_key for both passes: identical dropout masks, so the
            # penalty isolates the effect of the swapped immutable columns
            perm = jrand.permutation(perm_key, x.shape[0])
            x_tilde = jnp.where(self.invariance_mask.astype(bool), x[perm], x)
            y_tilde, _ = self.net.apply(params, apply_key, x_tilde, is_training=is_training)
            loss_5 = jnp.mean(optax.l2_loss(y_tilde, y_pred))
        else:
            # rows of the batch are independent (no batch norm), so the gradient
            # of the summed output w.r.t. the batch input stacks per-example grads
            grads = jax.grad(
                lambda xin: self.net.apply(params, apply_key, xin, is_training=is_training)[0].sum()
            )(x)
            loss_5 = ((grads * self.invariance_mask) ** 2).sum() / (
                x.shape[0] * self.invariance_mask.sum()
            )
        return self.configs.lambda_1 * loss_1 + self.configs.lambda_5 * loss_5


class InvariantActionabilityConfig(ActionabilityConfig, InvarianceConfig):
    """Both extensions at once: L4 on stage 2 and L5 on stage 1."""


class InvariantActionabilityModule(InvariantPredictorCounterNetModule,
                                   ActionabilityCounterNetModule):
    """Variant 6: incentive removal AND generator pressure.

        stage 1:  λ1·L1 + λ5·L5   (invariant predictor, from InvariantPredictor…)
        stage 2:  λ2·L2 + λ3·L3 + λ4·L4   (actionability loss, from Actionability…)

    The MRO composes the two overrides: `pred_loss_fn` comes from the invariance
    module, `exp_loss_fn` from the actionability module; both masks are built
    eagerly by the chained `init_net_opt` hooks.
    """


class RecourseAwareConfig(CounterNetConfig):
    """Config for the stage-1 recourse regularizer R_recourse (variant 8)."""

    lambda_r: float = Field(
        1.0, description="Weight of R_recourse in the stage-1 objective."
    )
    recourse_loss_kind: str = Field(
        "smoothed_hinge",
        description="'smoothed_hinge' (softplus(β(τ−g))/β on the logit margin; "
                    "flat on success, alive on failure) or 'quadratic' "
                    "(½(f−1)², the CounterNet-consistent ablation)."
    )
    recourse_margin_prob: float = Field(
        0.6, description="Margin τ expressed as a target-class probability; the "
                         "hinge is satisfied once f(x″) exceeds it."
    )
    recourse_beta: float = Field(
        2.0, description="Smoothing temperature β of the smoothed hinge."
    )


class RecourseAwarePredictorModule(CounterNetTrainingModule):
    """Variant 8: recourse regularizer R_recourse in stage 1 (working note §8).

        R_recourse = mean over eligible x (predicted unfavorable) of
                     ℓ_target( f_θ(x″), y_target = 1 )

    where x″ is the feasibility-projected CF proposed by the *current* generator
    (apply_constraints in soft mode during training — the same candidate the
    stage-2 losses see) and is treated as a FIXED candidate via stop-gradient:
    the stage-1 update moves only the predictor/encoder toward boundaries whose
    unfavorable decisions admit feasible recourse, and the generator receives no
    stage-1 gradient (working note §8.2's alternating prescription; stage 2 is
    unchanged and re-adapts the generator to each recourse-aware predictor).

    Under the fixed-candidate prescription, the cost term c(x, a) of the note's
    generic formulation is constant w.r.t. θ and drops out of the stage-1
    gradient — it matters only in an unrolled/bilevel variant.

    ℓ_target follows the surrogate analysis of the proposals page: the smoothed
    hinge on the logit margin (flat once the candidate crosses with margin τ, so
    the predictor is distorted only where recourse fails; non-vanishing repair
    gradient below the threshold), with the quadratic-on-probability form as the
    CounterNet-consistent ablation.
    """

    def __init__(self, config: RecourseAwareConfig | dict):
        config = (config if isinstance(config, RecourseAwareConfig)
                  else RecourseAwareConfig(**config))
        if config.recourse_loss_kind not in ("smoothed_hinge", "quadratic"):
            raise ValueError(
                "recourse_loss_kind must be 'smoothed_hinge' or 'quadratic', "
                f"got {config.recourse_loss_kind!r}"
            )
        if not 0.5 < config.recourse_margin_prob < 1.0:
            raise ValueError("recourse_margin_prob must lie in (0.5, 1).")
        super().__init__(config)

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        y_pred, cf_raw = self.net.apply(params, rng_key, x, is_training=is_training)
        cf = self._data_module.apply_constraints(x, cf_raw, hard=not is_training)
        # fixed candidate: no stage-1 gradient into the generator (nor through
        # the encoder's influence on the candidate) — working note §8.2
        cf = jax.lax.stop_gradient(cf)
        cf_y, _ = self.net.apply(params, rng_key, cf, is_training=is_training)

        loss_1 = self.loss_fn_1(y_pred, y)

        # eligible population: predicted-unfavorable instances; the indicator is
        # detached so eligibility does not itself receive gradient
        eligible = jax.lax.stop_gradient((y_pred < 0.5).astype(cf_y.dtype))
        p = jnp.clip(cf_y, 1e-6, 1 - 1e-6)
        if self.configs.recourse_loss_kind == "quadratic":
            per_example = optax.l2_loss(p, jnp.ones_like(p))
        else:
            g = jnp.log(p) - jnp.log1p(-p)  # logit of the target class
            tau = float(np.log(self.configs.recourse_margin_prob
                               / (1.0 - self.configs.recourse_margin_prob)))
            beta = self.configs.recourse_beta
            per_example = jax.nn.softplus(beta * (tau - g)) / beta
        r_recourse = ((eligible * per_example).sum()
                      / jnp.maximum(eligible.sum(), 1.0))
        return self.configs.lambda_1 * loss_1 + self.configs.lambda_r * r_recourse


class ExactRecourseConfig(RecourseAwareConfig):
    """Config for the exact R_recourse of the working note's formula (variant 8).

    R(θ) = E_{x∈ℰ}[ min_{a∈A(x)} { c(x,a) + ρ·ℓ_target(f_θ(T(x,a))) } ]

    No immutability projection: T(x,a) = apply_constraints(x, a, hard=False)
    applies only the encoded-domain rules (categorical softmax, continuous clip).
    """

    rho: float = Field(
        1.0, description="Weight ρ of ℓ_target against the cost c inside the min."
    )
    inner_steps: int = Field(
        10, description="Gradient steps of the per-instance inner minimization."
    )
    inner_lr: float = Field(
        0.5, description="Step size of the inner minimization (logit space)."
    )


class ExactRecoursePredictorModule(CounterNetTrainingModule):
    """Variant 8: the working note's R_recourse formula, implemented literally.

    Per predictor update:
      - eligible population ℰ: predicted-unfavorable instances (detached mask);
      - action a: a raw proposal vector; A(x) is unrestricted, admissibility is
        enforced by T(x,a) = apply_constraints(x, a, hard=False) — domain rules
        only, NO immutable-feature copying;
      - inner min: `inner_steps` gradient-descent steps on a, initialized at the
        null action (a₀ reproduces x through T), against frozen parameters;
      - cost c(x,a) = mean over columns of ½(T(x,a) − x)² (the L3 convention);
      - ℓ_target: smoothed hinge on the logit margin (or quadratic ablation);
      - envelope (Danskin) gradient: the minimizer a* is detached and the outer
        objective c(x,a*) + ρ·ℓ_target(f_θ(T(x,a*))) is differentiated w.r.t. θ
        — the θ-gradient flows through ℓ_target only (c is θ-free at fixed a*),
        which is the exact gradient of the min under a unique minimizer.

    Unlike the fixed-candidate form (8a), the inner search re-solves against the
    *current* predictor every step, so stage 1 cannot outrun a slow generator.
    The set-emptying degenerate optimum (deny nobody ⇒ ℰ = ∅ ⇒ R = 0) remains
    expressible, since ℰ is defined by the model's own predictions.
    """

    def __init__(self, config: ExactRecourseConfig | dict):
        config = (config if isinstance(config, ExactRecourseConfig)
                  else ExactRecourseConfig(**config))
        if config.recourse_loss_kind not in ("smoothed_hinge", "quadratic"):
            raise ValueError(
                f"recourse_loss_kind must be 'smoothed_hinge' or 'quadratic', "
                f"got {config.recourse_loss_kind!r}"
            )
        super().__init__(config)
        self._categorical_mask = None

    def init_net_opt(self, data_module, key):
        out = super().init_net_opt(data_module, key)
        _ = self.categorical_mask
        return out

    @property
    def categorical_mask(self) -> jnp.ndarray:
        if self._categorical_mask is None:
            names = [f.name for f in self._data_module.features if f.is_categorical]
            self._categorical_mask = feature_col_mask(self._data_module, names)
        return self._categorical_mask

    def _ell_target(self, p):
        """Target loss on the favorable-class probability of the reached state."""
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        if self.configs.recourse_loss_kind == "quadratic":
            return optax.l2_loss(p, jnp.ones_like(p))
        g = jnp.log(p) - jnp.log1p(-p)
        tau = float(np.log(self.configs.recourse_margin_prob
                           / (1.0 - self.configs.recourse_margin_prob)))
        return jax.nn.softplus(self.configs.recourse_beta * (tau - g)) \
            / self.configs.recourse_beta

    @partial(jax.jit, static_argnames=["self", "is_training"])
    def pred_loss_fn(self, params, rng_key, batch, is_training: bool = True):
        x, y = batch
        y_pred, _ = self.net.apply(params, rng_key, x, is_training=is_training)
        loss_1 = self.loss_fn_1(y_pred, y)
        eligible = jax.lax.stop_gradient((y_pred < 0.5).astype(y_pred.dtype))

        def inner_obj(a, params_):
            """Per-example c(x,a) + ρ·ℓ_target(f(T(x,a))), shape (N, 1)."""
            s = self._data_module.apply_constraints(x, a, hard=False)
            cost = 0.5 * ((s - x) ** 2).mean(axis=1, keepdims=True)
            f_s, _ = self.net.apply(params_, rng_key, s, is_training=is_training)
            return cost + self.configs.rho * self._ell_target(f_s)

        # inner min over actions against frozen parameters. a₀ is the null
        # action: sharp logits on categorical columns so softmax(a₀) ≈ x, raw
        # values on continuous columns so clip(a₀) = x.
        params_frozen = jax.lax.stop_gradient(params)
        a = jnp.where(self.categorical_mask.astype(bool), 10.0 * x, x)
        for _ in range(self.configs.inner_steps):
            g_a = jax.grad(lambda aa: inner_obj(aa, params_frozen).sum())(a)
            a = a - self.configs.inner_lr * g_a
        a_star = jax.lax.stop_gradient(a)

        # envelope: evaluate at the detached minimizer with live parameters
        per_example = inner_obj(a_star, params)
        r_recourse = ((eligible * per_example).sum()
                      / jnp.maximum(eligible.sum(), 1.0))
        return self.configs.lambda_1 * loss_1 + self.configs.lambda_r * r_recourse


def get_or_train(name: str, dm, cfnet: CounterNet) -> CounterNet:
    """Load cached params for `name` if present, otherwise train and cache."""
    variant_dir = OUT_DIR / name
    params_path = variant_dir / "params.pkl"
    if params_path.exists():
        print(f"[{name}] loading cached model from {params_path}")
        cfnet.set_data_module(dm)
        cfnet.module = cfnet._init_model(cfnet.config, cfnet.module)
        cfnet.module._data_module = dm
        with open(params_path, "rb") as f:
            cfnet.params = pickle.load(f)
        cfnet._is_trained = True
        return cfnet
    print(f"[{name}] training for {EPOCHS} epochs with seed {TRAIN_SEED}...")
    start = time.time()
    cfnet.train(dm, batch_size=BATCH_SIZE, epochs=EPOCHS, seed=TRAIN_SEED)
    print(f"[{name}] training finished in {time.time() - start:.1f}s")
    variant_dir.mkdir(parents=True, exist_ok=True)
    with open(params_path, "wb") as f:
        pickle.dump(cfnet.params, f)
    return cfnet


def _predictor_l4(cfnet, dm, xs, loss_L3: float) -> float:
    """Value of the experiment's stage-1 regularizer on the test split."""
    module = cfnet.module
    if not hasattr(module, "invariance_mask"):
        return loss_L3                      # proximity form (E1-E4.1 and references)
    m = module.invariance_mask
    grads = jax.grad(
        lambda xin: module.net.apply(
            cfnet.params, jrand.PRNGKey(0), xin, is_training=False)[0].sum()
    )(xs)
    return float(((grads * m) ** 2).sum() / (xs.shape[0] * m.sum()))


def evaluate_variant(cfnet: CounterNet, dm) -> dict:
    """Evaluate on the held-out test split only (hard CEs)."""
    xs, ys = dm["test"]
    d = xs.shape[1]
    cfs = cfnet.module.generate_cf(cfnet.params, jrand.PRNGKey(0), xs)
    preds = cfnet.pred_fn(xs)
    mask = feature_col_mask(dm, IMMUTABLE)
    n_mutable_cols = d - int(mask.sum())

    # predictor invariance gap: mean |ŷ(x) − ŷ(x̃)| where x̃ carries the immutable
    # block of another test instance (fixed permutation) — the demographic
    # sensitivity of the predictor, i.e. the incentive the generator can exploit
    perm = jrand.permutation(jrand.PRNGKey(1), xs.shape[0])
    x_swap = jnp.where(mask.astype(bool), xs[perm], xs)
    pred_gap = float(jnp.abs(preds[:, 1] - cfnet.pred_fn(x_swap)[:, 1]).mean())

    # margin metric (working note §7.4): mean predicted favorable-class
    # probability of the CE for originally-denied instances. Binary validity
    # saturates near 1.0; this captures confidence/margin movements it hides.
    denied = np.asarray(jnp.argmax(preds, axis=-1) == 0)
    cf_preds = cfnet.pred_fn(cfs)
    cf_target_conf = float(cf_preds[denied, 1].mean()) if denied.any() else float("nan")

    # test-split values of the training losses, evaluated on the delivered (hard)
    # CEs. L4 (the stage-1 recourse-cost term) is the same scalar as L3 by
    # construction — the two differ only in which stage's gradient they feed.
    p1 = preds[:, 1:2]
    y_true = ys.reshape(-1, 1).astype(p1.dtype)
    loss_L1 = float(jnp.mean(optax.l2_loss(p1, y_true)))
    y_prime = 1.0 - jnp.round(p1)
    loss_L2 = float(jnp.mean(optax.l2_loss(cf_preds[:, 1:2], y_prime)))
    loss_L3 = float(jnp.mean(optax.l2_loss(cfs, xs)))

    row = {
        "n_test": xs.shape[0],
        "acc": float((jnp.argmax(preds, axis=-1) == ys.reshape(-1)).mean()),
        "pred_gap": pred_gap,
        "validity": float(compute_validity(xs, cfs, cfnet.pred_fn)),
        "cf_target_conf": cf_target_conf,
        "L1": loss_L1,
        "L2": loss_L2,
        "L3": loss_L3,
        # L4 is the predictor-side regularizer, whose definition is per-experiment:
        # the proximity form for the E1–E4.1 family (identical to L3), the masked
        # input-gradient penalty for E5. Units therefore differ between the two.
        "L4": _predictor_l4(cfnet, dm, xs, loss_L3),
        "proximity": float(compute_proximity(xs, cfs)),
    }
    row["prox_norm"] = row["proximity"] / d
    # the paper's sparsity metric (Table 3 "Spar."): per-feature mean of changed
    # one-hot columns, averaged over features; lower = sparser
    row["sparsity"] = float(compute_sparsity(xs, cfs, dm.features.feature_indices))
    # proximity over mutable columns only: the fair "user effort" comparison,
    # unaffected by how many columns a mechanism freezes
    prox_mutable = float(jnp.abs((xs - cfs) * (1 - mask)).sum(axis=1).mean())
    row["proximity_mutable"] = prox_mutable
    row["prox_mutable_norm"] = prox_mutable / n_mutable_cols

    idx = {f.name: rng for f, rng in zip(dm.features, dm.features.feature_indices)}
    any_changed = jnp.zeros(xs.shape[0], dtype=bool)
    for name in IMMUTABLE:
        s, e = idx[name]
        changed = jnp.abs(xs[:, s:e] - cfs[:, s:e]).sum(axis=1) > 1e-6
        row[f"viol_{name}"] = float(changed.mean())
        any_changed |= changed
    row["viol_any"] = float(any_changed.mean())

    # mean absolute change of age (continuous, [0,1] scale) — the "costly" feature
    # soft-penalized in the mixed variant; reported for every row for comparison
    s, e = idx[AGE]
    row["chg_age"] = float(jnp.abs(xs[:, s:e] - cfs[:, s:e]).sum(axis=1).mean())
    age_decrease = (cfs[:, s:e] < xs[:, s:e] - 1e-6).any(axis=1)
    row["viol_age_decrease"] = float(age_decrease.mean())
    row["max_age_decrease"] = float(jnp.maximum(xs[:, s:e] - cfs[:, s:e], 0).max())

    # Every categorical group must be exactly one of the data module's known
    # one-hot categories. Separately verify the two Adult 'Other/Unknown' bans.
    invalid_categorical = jnp.zeros(xs.shape[0], dtype=bool)
    disallowed_any = jnp.zeros(xs.shape[0], dtype=bool)
    for feat, (start, end) in zip(dm.features, dm.features.feature_indices):
        if not feat.is_categorical:
            continue
        group = cfs[:, start:end]
        is_binary = jnp.logical_or(
            jnp.isclose(group, 0.0, atol=1e-6),
            jnp.isclose(group, 1.0, atol=1e-6),
        ).all(axis=1)
        has_one_category = jnp.isclose(group.sum(axis=1), 1.0, atol=1e-6)
        invalid_categorical |= ~(is_binary & has_one_category)

        categories = list(feat.transformation.transformer.categories_)
        for label in DISALLOWED_MUTABLE_CATEGORIES.get(feat.name, []):
            selected = group[:, categories.index(label)] > 0.5
            row[f"out_{feat.name}_other_unknown"] = float(selected.mean())
            disallowed_any |= selected
    row["invalid_categorical_any"] = float(invalid_categorical.mean())
    row["out_disallowed_any"] = float(disallowed_any.mean())

    # race='Other' is a legitimate immutable source value, not a banned generated
    # category. Report its stratum explicitly to guard against conflating the labels.
    # A full pass, not next(): relax's FeaturesList is its own iterator, so a partial
    # pass would leave later loops over dm.features starting after "race".
    race_feat = [feat for feat in dm.features if feat.name == "race"][0]
    race_start, race_end = idx["race"]
    race_categories = list(race_feat.transformation.transformer.categories_)
    race_other = xs[:, race_start + race_categories.index("Other")] > 0.5
    race_changed = jnp.abs(
        xs[:, race_start:race_end] - cfs[:, race_start:race_end]).sum(axis=1) > 1e-6
    n_race_other = int(race_other.sum())
    row["n_race_other"] = n_race_other
    row["viol_race_other"] = float(
        (race_changed & race_other).sum() / max(n_race_other, 1))
    return row


def main(lambdas: list[float], age_lambdas: list[float], invariance_lambdas: list[float],
         combo_lambdas: list[float], recourse_lambdas: list[float],
         predictor_proximity_lambdas: list[float], scheduled_lambdas: list[float],
         finetune_lambdas: list[float], finetune_every_k: list[int],
         gradient_lambdas: list[float]):
    records = {}

    # --- Variant 1: projection (paper's mechanism, extended to marital_status) ---
    dm_proj = load_dm(IMMUTABLE)
    cfnet_proj = get_or_train("projection", dm_proj, CounterNet())
    records["projection"] = evaluate_variant(cfnet_proj, dm_proj)

    # --- Variant 7: reachability projection. Keep variant 1's three exact
    #     immutable copies; require normalized age_cf >= age_x; and remove the
    #     explicit 'Other/Unknown' outcomes from mutable workclass/occupation.
    dm_reach = load_dm(IMMUTABLE)
    config_reach = ReachabilityProjectionConfig(
        fixed_features=IMMUTABLE,
        nondecreasing_features=[AGE],
        disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
    )
    module_reach = ReachabilityProjectionModule(config_reach)
    cfnet_reach = get_or_train(
        "reachability_projection_fixed3_age_nondec_no_unknown",
        dm_reach,
        CounterNet(config_reach, cfnet_module=module_reach),
    )
    records["7: reachability projection (fixed 3, age non-decreasing, no mutable unknown)"] = \
        evaluate_variant(cfnet_reach, dm_reach)

    # --- Actionability-recourse experiment (its own report page): variant 7's
    #     reachability projection in the CF network, plus L3's proximity form
    #     added to the predictor objective as L4 (stage 1), swept over λ4.
    #     Gradient flows through the candidate into Encoder/Predictor; the
    #     Explainer's weights are stop-gradiented (stage-2 property). ---
    for l4p in predictor_proximity_lambdas:
        dm_ar = load_dm(IMMUTABLE)
        config_ar = PredictorProximityConfig(
            fixed_features=IMMUTABLE,
            nondecreasing_features=[AGE],
            disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
            lambda_4=l4p,
        )
        module_ar = PredictorProximityReachabilityModule(config_ar)
        cfnet_ar = get_or_train(
            f"actionability_recourse_predprox_l4_{l4p}",
            dm_ar,
            CounterNet(config_ar, cfnet_module=module_ar),
        )
        records[f"AR: v7 reachability + predictor proximity L4 (λ4={l4p:g})"] = \
            evaluate_variant(cfnet_ar, dm_ar)

    # --- Scheduled L4: same setup, but L4 is charged intermittently ('every_k')
    #     or only after a warm-up ('after_k'). Both target the accumulation
    #     mechanism — L4's gain is re-equilibrated by stage 2 each step while its
    #     representation distortion accumulates. Swept at the λ4 values where the
    #     always-on schedule failed. ---
    steps_per_epoch = int(np.ceil(len(load_dm([])["train"][0]) / BATCH_SIZE))
    for schedule, k, tag in [("every_k", 10, "every 10 steps"),
                             ("every_k", 20, "every 20 steps"),
                             ("after_k", 50 * steps_per_epoch, "after 50 epochs"),
                             ("after_k", 20 * steps_per_epoch, "after 20 epochs")]:
        for l4p in scheduled_lambdas:
            dm_s = load_dm(IMMUTABLE)
            config_s = ScheduledPredictorProximityConfig(
                fixed_features=IMMUTABLE,
                nondecreasing_features=[AGE],
                disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
                lambda_4=l4p, l4_schedule=schedule, l4_k=k,
            )
            module_s = ScheduledPredictorProximityModule(config_s)
            cfnet_s = get_or_train(
                f"actionability_recourse_predprox_{schedule}_{k}_l4_{l4p}",
                dm_s,
                CounterNet(config_s, cfnet_module=module_s),
            )
            records[f"AR: predictor L4 {tag} (λ4={l4p:g})"] = \
                evaluate_variant(cfnet_s, dm_s)

    # --- Two-phase: joint CounterNet training with no L4 (phase 1 = the
    #     reachability-projection reference model), then fine-tune ONLY the
    #     predictor on λ1·L1 + λ4·L4 with the CF generator frozen and stage 2
    #     switched off entirely (phase 2). Removes stage-2 re-equilibration, so
    #     it isolates what the stage-1 channel can achieve unopposed. ---
    for l4p in finetune_lambdas:
        name_ft = f"actionability_recourse_finetune_l4_{l4p}_{FINETUNE_EPOCHS}ep"
        path_ft = OUT_DIR / name_ft / "params.pkl"
        dm_ft = load_dm(IMMUTABLE)
        config_ft = ReachabilityProjectionConfig(
            fixed_features=IMMUTABLE,
            nondecreasing_features=[AGE],
            disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
        )
        cfnet_ft = CounterNet(
            config_ft, cfnet_module=ReachabilityProjectionModule(config_ft))
        if path_ft.exists():
            print(f"[{name_ft}] loading cached model from {path_ft}")
            cfnet_ft.set_data_module(dm_ft)
            cfnet_ft.module = cfnet_ft._init_model(cfnet_ft.config, cfnet_ft.module)
            cfnet_ft.module._data_module = dm_ft
            with open(path_ft, "rb") as f:
                cfnet_ft.params = pickle.load(f)
            cfnet_ft._is_trained = True
        else:
            # phase 1: reuse the jointly-trained, no-L4 model
            cfnet_ft = get_or_train(
                "reachability_projection_fixed3_age_nondec_no_unknown",
                dm_ft, cfnet_ft)
            print(f"[{name_ft}] phase 2: fine-tuning predictor for "
                  f"{FINETUNE_EPOCHS} epochs (λ4={l4p:g}, generator frozen)...")
            start = time.time()
            cfnet_ft = finetune_predictor_on_l4(
                cfnet_ft, dm_ft, lambda_4=l4p, epochs=FINETUNE_EPOCHS)
            print(f"[{name_ft}] fine-tuning finished in {time.time() - start:.1f}s")
            (OUT_DIR / name_ft).mkdir(parents=True, exist_ok=True)
            with open(path_ft, "wb") as f:
                pickle.dump(cfnet_ft.params, f)
        records[f"AR: two-phase, fine-tune predictor on L4 (λ4={l4p:g})"] = \
            evaluate_variant(cfnet_ft, dm_ft)

    # --- Experiment 5: L4 becomes a masked input-gradient penalty on the
    #     predictor — local insensitivity to the immutable coordinates — with the
    #     reachability projection and stage 2 unchanged. ---
    for l4g in gradient_lambdas:
        dm_g = load_dm(IMMUTABLE)
        config_g = ImmutableGradientConfig(
            fixed_features=IMMUTABLE,
            nondecreasing_features=[AGE],
            disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
            invariant_features=IMMUTABLE, lambda_4=l4g,
        )
        module_g = ImmutableGradientPredictorModule(config_g)
        cfnet_g = get_or_train(
            f"actionability_recourse_gradpen_l4_{l4g}", dm_g,
            CounterNet(config_g, cfnet_module=module_g))
        records[f"AR: predictor immutable-gradient penalty (λ4={l4g:g})"] = \
            evaluate_variant(cfnet_g, dm_g)

    # --- Experiment 4.1: the two-phase fine-tune, but with L4 charged only on
    #     every k-th step of phase 2 (E2's intermittent gating applied inside the
    #     fine-tuning phase). Unlike E2 there is no stage 2 to recover in between,
    #     so this scales the total charge rather than granting recovery. ---
    for k_ft in finetune_every_k:
        for l4p in scheduled_lambdas:
            name_k = (f"actionability_recourse_finetune_everyk{k_ft}"
                      f"_l4_{l4p}_{FINETUNE_EPOCHS}ep")
            path_k = OUT_DIR / name_k / "params.pkl"
            dm_k = load_dm(IMMUTABLE)
            config_k = ReachabilityProjectionConfig(
                fixed_features=IMMUTABLE,
                nondecreasing_features=[AGE],
                disallowed_categories=DISALLOWED_MUTABLE_CATEGORIES,
            )
            cfnet_k = CounterNet(
                config_k, cfnet_module=ReachabilityProjectionModule(config_k))
            if path_k.exists():
                print(f"[{name_k}] loading cached model from {path_k}")
                cfnet_k.set_data_module(dm_k)
                cfnet_k.module = cfnet_k._init_model(cfnet_k.config, cfnet_k.module)
                cfnet_k.module._data_module = dm_k
                with open(path_k, "rb") as f:
                    cfnet_k.params = pickle.load(f)
                cfnet_k._is_trained = True
            else:
                cfnet_k = get_or_train(
                    "reachability_projection_fixed3_age_nondec_no_unknown",
                    dm_k, cfnet_k)
                print(f"[{name_k}] phase 2: fine-tuning {FINETUNE_EPOCHS} epochs, "
                      f"L4 every {k_ft} steps (λ4={l4p:g}), generator frozen...")
                start = time.time()
                cfnet_k = finetune_predictor_on_l4(
                    cfnet_k, dm_k, lambda_4=l4p, epochs=FINETUNE_EPOCHS,
                    every_k=k_ft)
                print(f"[{name_k}] fine-tuning finished in {time.time() - start:.1f}s")
                (OUT_DIR / name_k).mkdir(parents=True, exist_ok=True)
                with open(path_k, "wb") as f:
                    pickle.dump(cfnet_k.params, f)
            records[f"AR: two-phase fine-tune, L4 every {k_ft} steps (λ4={l4p:g})"] = \
                evaluate_variant(cfnet_k, dm_k)

    # --- Reference: feasibility CounterNet (paper §4.2 / jax-relax default:
    #     race + gender projected, marital_status free) ---
    dm_feas = load_dm(["race", "gender"])
    cfnet_feas = get_or_train("feasibility_race_gender", dm_feas, CounterNet())
    records["feasibility CounterNet (race+gender projected)"] = evaluate_variant(cfnet_feas, dm_feas)

    # --- Variant 2: actionability loss, swept over lambda_4 ---
    for l4 in lambdas:
        dm_act = load_dm([])
        config = ActionabilityConfig(penalized_features=IMMUTABLE, lambda_4=l4)
        module = ActionabilityCounterNetModule(config)
        cfnet_act = get_or_train(f"actionability_l4_{l4}", dm_act,
                                 CounterNet(config, cfnet_module=module))
        label = ("base CounterNet (paper config: no enforcement)" if l4 == 0
                 else f"actionability λ4={l4:g}")
        records[label] = evaluate_variant(cfnet_act, dm_act)

    # --- Variant 3: mixed — project {race, gender, marital_status} AND
    #     soft-penalize age (costly but changeable), on disjoint feature sets.
    #     Swept over lambda_4 to map the age-cost vs validity trade-off. ---
    for l4 in age_lambdas:
        dm_mix = load_dm(IMMUTABLE)
        config_mix = ActionabilityConfig(penalized_features=["age"], lambda_4=l4)
        module_mix = ActionabilityCounterNetModule(config_mix)
        cfnet_mix = get_or_train(f"mixed_proj3_age_l4_{l4}", dm_mix,
                                 CounterNet(config_mix, cfnet_module=module_mix))
        records[f"mixed: project 3 + penalize age (λ4={l4:g})"] = evaluate_variant(cfnet_mix, dm_mix)

    # --- Variant 4 (4a): variant 1's weights, inference WITHOUT the immutability
    #     projection. Training is identical to variant 1 (projection precedes every
    #     loss), so the cached weights are reused; only the inference pipeline differs:
    #     the data module hooked at generation time has no is_immutable flags, so
    #     apply_constraints performs argmax/clip but never copies columns from x.
    #     Measures the generator's raw, unsupervised outputs on the immutable columns.
    assert (OUT_DIR / "projection" / "params.pkl").exists(), \
        "variant 4 reuses variant 1's weights; run variant 1 first"
    dm_free = load_dm([])
    cfnet_v4 = get_or_train("projection", dm_free, CounterNet())
    records["variant 4: v1 weights, no inference projection"] = evaluate_variant(cfnet_v4, dm_free)

    # --- Variant 5: predictor-invariance regularizer L5 (stage 1), no projection,
    #     no L4. Removes the generator's *incentive* to change immutable features
    #     instead of fighting its output. A = consistency, B = gradient penalty. ---
    for kind, tag in [("consistency", "A"), ("gradient", "B")]:
        for l5 in invariance_lambdas:
            dm_inv = load_dm([])
            config_inv = InvarianceConfig(
                invariant_features=IMMUTABLE, lambda_5=l5, invariance_kind=kind)
            module_inv = InvariantPredictorCounterNetModule(config_inv)
            cfnet_inv = get_or_train(f"invariance_{kind}_l5_{l5}", dm_inv,
                                     CounterNet(config_inv, cfnet_module=module_inv))
            records[f"5{tag}: invariant predictor ({kind}, λ5={l5:g})"] = \
                evaluate_variant(cfnet_inv, dm_inv)

    # --- Variant 6: L5 (stage 1, consistency, λ5 = 10) AND L4 (stage 2), swept
    #     over lambda_4. Tests whether generator pressure works better once the
    #     predictor's demographic incentive is gone. ---
    for l4 in combo_lambdas:
        dm_c = load_dm([])
        config_c = InvariantActionabilityConfig(
            penalized_features=IMMUTABLE, lambda_4=l4,
            invariant_features=IMMUTABLE, lambda_5=10.0, invariance_kind="consistency")
        module_c = InvariantActionabilityModule(config_c)
        cfnet_c = get_or_train(f"combined_consistency_l5_10.0_l4_{l4}", dm_c,
                               CounterNet(config_c, cfnet_module=module_c))
        records[f"6: invariant pred. (A, λ5=10) + L4 (λ4={l4:g})"] = \
            evaluate_variant(cfnet_c, dm_c)

    # --- Variant 8a (superseded operationalization): fixed generator candidate,
    #     variant-1 projection, prediction-defined eligibility. Kept for the
    #     record; collapsed in both pre-registered ways. ---
    for kind, tag, lams in [("smoothed_hinge", "hinge", recourse_lambdas),
                            ("quadratic", "quadratic", [1.0])]:
        for lr_ in lams:
            dm_r = load_dm(IMMUTABLE)
            config_r = RecourseAwareConfig(lambda_r=lr_, recourse_loss_kind=kind)
            module_r = RecourseAwarePredictorModule(config_r)
            cfnet_r = get_or_train(f"recourse_aware_{kind}_lr_{lr_}", dm_r,
                                   CounterNet(config_r, cfnet_module=module_r))
            records[f"8a (superseded): fixed-candidate, projected ({tag}, λR={lr_:g})"] = \
                evaluate_variant(cfnet_r, dm_r)

    # --- Variant 8: the working note's R_recourse formula, literal form.
    #     Explicit inner min over actions (k gradient steps per instance,
    #     Danskin envelope gradient), cost inside the min weighted against
    #     ℓ_target by ρ, and NO immutability projection: T applies only the
    #     encoded-domain rules. ---
    for lr_ in recourse_lambdas:
        dm_e = load_dm([])
        config_e = ExactRecourseConfig(lambda_r=lr_, recourse_loss_kind="smoothed_hinge")
        module_e = ExactRecoursePredictorModule(config_e)
        cfnet_e = get_or_train(f"recourse_exact_hinge_lr_{lr_}", dm_e,
                               CounterNet(config_e, cfnet_module=module_e))
        records[f"8: exact R_recourse (inner min, no projection, λR={lr_:g})"] = \
            evaluate_variant(cfnet_e, dm_e)

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "variant"
    print("\n" + df.to_string(float_format=lambda v: f"{v:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "results.csv")
    print(f"\nSaved results to {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 5.0, 20.0],
                        help="lambda_4 values for the actionability sweep (variant 2)")
    parser.add_argument("--age-lambdas", type=float, nargs="+",
                        default=[0.05, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0],
                        help="lambda_4 values for the mixed variant's age penalty (variant 3)")
    parser.add_argument("--invariance-lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0],
                        help="lambda_5 values for the predictor-invariance sweep (variant 5, A and B)")
    parser.add_argument("--combo-lambdas", type=float, nargs="+",
                        default=[1.0, 20.0, 1000.0],
                        help="lambda_4 values for variant 6 (invariant predictor A λ5=10 + L4)")
    parser.add_argument("--recourse-lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0],
                        help="lambda_R values for variant 8's smoothed-hinge sweep "
                             "(the quadratic ablation runs at λR=1 only)")
    parser.add_argument("--predictor-proximity-lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0],
                        help="lambda_4 values for the actionability-recourse experiment "
                             "(v7 reachability + stage-1 proximity on the predictor)")
    parser.add_argument("--scheduled-lambdas", type=float, nargs="+",
                        default=[1.0, 10.0],
                        help="lambda_4 values for the scheduled-L4 runs (every_k / after_k)")
    parser.add_argument("--finetune-lambdas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0],
                        help="lambda_4 values for the two-phase fine-tuning experiment")
    parser.add_argument("--finetune-every-k", type=int, nargs="+", default=[10, 20],
                        help="step periods for experiment 4.1 (L4 gated inside phase 2)")
    parser.add_argument("--gradient-lambdas", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                        help="lambda_4 values for experiment 5 (immutable input-gradient penalty)")
    args = parser.parse_args()
    main(args.lambdas, args.age_lambdas, args.invariance_lambdas, args.combo_lambdas,
         args.recourse_lambdas, args.predictor_proximity_lambdas, args.scheduled_lambdas,
         args.finetune_lambdas, args.finetune_every_k, args.gradient_lambdas)
