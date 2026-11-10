# Model-approx CF: C1 and C2, with loss-only and projected extensions

These experiments compare C1/C2 conditional VAE Generators under the saved
CounterNet Adult data protocol: 36 original/sweep runs and six earlier projected
reference runs (42 trained Generators in total, including 24 new sweep fits). This does
not reproduce the paper's filtered Adult
experiment and does not train or modify CounterNet itself.

The paper studies both **C1** (age cannot decrease) and **C2** (education cannot
decrease; higher education requires higher age; unchanged education permits only
nondecreasing age). C1 and C2 are separate settings, but C2 retains the age rule.
Each setting has an original, a loss-only extension and a projected extension,
with three matched seeds for each variant.
The projection policy is identical across settings and does not add hard
education or strict age–education coupling rules. Its compliance is therefore
reported separately from full C2 compliance.

Method: Mahajan, Tan and Sharma, *Preserving Causal Constraints in Counterfactual
Explanations for Machine Learning Classifiers*, arXiv:1912.03277.
Reference code: https://github.com/divyat09/cf-feasibility at
`db715fa107cc1ca328ae6d7863de8a752ca13bdc`.

## Credit and OULAD adaptations

The additional workflow fits **72 Generators**: two datasets × two adapted
settings × six coefficients (0, 0.1, 0.3, 1, 3, 10) × three seeds. Alpha=0 is the
adapted baseline; all nonzero alphas add the full dataset-specific CounterNet
constraints through penalties, with no projection.

The exporter loads the exact saved feature transformations, category order and
train/test indices from `relax-assets/{credit,oulad}/data/`. Credit has 22,500
training and 7,500 test rows, with 33 encoded inputs. OULAD has 24,444 training
and 8,149 test rows, with 127 encoded inputs. No data refitting, row filtering,
category merging or replacement split is performed. Each dataset gets a separate
Model-approx classifier trained on its original training rows and then frozen
for all its Generator runs; this is not the trained CounterNet Predictor.

C1* and C2* deliberately mean **dataset adaptations**, not published baselines:

- Credit C1*: original numerical age-decrease hinge on AGE.
- OULAD C1*: sum the raw scores for age bands below the factual age band, in
  the order `0-35 < 35-55 < 55<=`. No age band is masked or projected.
- Credit C2*: retain age and add an education incentive using documented ranks
  `3 (high school) -> 0`, `2 (university) -> 1`, `1 (graduate school) -> 2`.
  Codes 0/4/5/6 are unranked. With K={3,2,1}, use
  `D = mean[I(factual education in K) * sum_(k in K) r_k*(rank(k)-rank(x))]`.
  Unranked factual categories contribute zero; no rows are removed. An unranked
  generated category contributes no education comparison. This is partial
  ordering, not an invented rank for unknown/other education.
- OULAD C2*: retain ordinal age and the native linear education-score incentive,
  using No Formal quals=0, Lower Than A Level=1, A Level or Equivalent=2,
  HE Qualification=3, Post Graduate Qualification=4. Its age bands cannot show
  a strict increase in actual age within one band. Neither C2* soft objective
  guarantees the original verbal age/education coupling.

The extra Credit penalties preserve SEX/MARRIAGE and discourage EDUCATION
0/5/6 (code 4 stays allowed). The extra OULAD penalties preserve gender/region/
disability and discourage imd_band='0'. They are the same CounterNet policies,
expressed as absolute coordinate losses and one-sided score penalties. Age is
already in each baseline and is not added twice.

C1* transfers margin=.764, validity weight=29, age weight=192; C2* transfers
margin=.344, validity weight=76, age/education weight=87 and education factor=.05.
Only the added loss receives alpha. The native raw-range-weighted reconstruction
and final-category omission are retained: the omitted coordinate is MARRIAGE_3
for Credit and disability_Y for OULAD. Added immutability penalties and reported
proximity include those coordinates. Native coefficients were tuned for Adult,
not these datasets; large financial/click ranges can dominate reconstruction.
The report shows actual training loss magnitudes and does not claim optimal
Model-approx performance or automatic cross-dataset generalization.

Generator architecture, 50 epochs, 50 draws per training input, ten per test input,
seeds, initialization and batch/evaluation randomness are matched across each
comparison. Data-dependent input/output widths change; hidden/latent widths and
other settings match Adult. The mathematical losses are vectorized to avoid
repeated full-input gradient allocations. `vectorization_check.json` records
loss/gradient comparisons for all 24 dataset/setting/coefficient combinations.
The pre-vectorization pilot runs are archived separately and excluded; every
reported Generator is trained under the same vectorized implementation.

New files:

- `cf_feasibility_multidata_prepare.py`: exact saved data export and source hashes.
- `cf_feasibility_multidata.py`: converted losses, frozen classifiers, 72 fits.
- `cf_feasibility_multidata_results.py`: all metrics, paired differences and an
  independent audit of 5,633,640 decoded samples with exact raw replay.
- `cf_feasibility_multidata_report.py`: report tables, formulas and limitations.
- `cf_feasibility_multidata_availability.py`: separate affine-classifier witnesses
  under the encoded CounterNet policy, with no distance budget or full C2
  coupling; never substituted for Generator outputs.
- `test_cf_feasibility_multidata.py`: source-loss parity, ordinal/education cases,
  saved splits, and complete training/evaluation smoke checks.
- `trained_models/cf_feasibility_multidata/`: per-dataset data/protocol/classifier,
  `c1|c2/alpha_*/seed_*/` checkpoints, histories, candidates and evaluations;
  root `summary.json`, `results.csv`, `audit.json` and equivalence record.

```sh
.venv/bin/python cf_feasibility_multidata_prepare.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_multidata -v
.venv-cf-feasibility/bin/python cf_feasibility_multidata.py all --dataset credit
.venv-cf-feasibility/bin/python cf_feasibility_multidata.py all --dataset oulad
.venv-cf-feasibility/bin/python cf_feasibility_multidata_results.py all
.venv-cf-feasibility/bin/python cf_feasibility_multidata_availability.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py
```

Training also accepts `--setting c1|c2`, `--alpha` and `--seed`. The two datasets
and settings can run in independent CPU processes after training each classifier
once with the `classifier` command. Every training worker uses one Torch thread
and deterministic algorithms. Saved complete runs are reused; incomplete runs
restart from their matched initial seed and full budget.

## Five-strength sensitivity sweep

The prespecified grid is `alpha = 0.1, 0.3, 1, 3, 10` for both C1 and C2, with
Generator seeds 42, 43 and 44. The six existing alpha=1 runs and six original
alpha=0 controls are reused read-only. Four additional weights per setting give
24 new fits, each using the same 50-epoch training protocol. No policy projection
is used. Only the coefficient of the **added** penalties changes:

```text
C1: L_base + 29 H_0.764 + 192 A + (alpha * 192) P_extra
C2: L_base + 76 H_0.344 + 87 A - (87 * 0.05) D + (alpha * 87) P_extra
```

The original validity margins/weights, age coefficients and education term remain
fixed. C1's added weights are 19.2, 57.6, 192, 576 and 1920; C2's are 8.7, 26.1,
87, 261 and 870. Alpha=1 matches the prior loss and gradients exactly. All seeds,
initializations, batch ordering, evaluation noise, architecture, data and shared
classifier are unchanged. Standard decoding remains categorical argmax plus
numeric bounds; it does not enforce any of the policy constraints.

Artifacts live in `trained_models/cf_feasibility_penalty_sweep/`:

- `c1/alpha_*/` and `c2/alpha_*/`: immutable protocols, shared data/classifier
  copies, and per-seed checkpoints, epoch histories and evaluations.
- `summary.json` and `results.csv`: all 36 runs and 108 prediction-direction
  cohort rows; CounterNet-style metrics, seed means/SDs and paired differences.
- `audit.json`: exact replay of 2,930,760 raw draws, independent decoding and
  metric checks, unchanged reused checkpoints/candidates, coefficient checks in
  every recorded training epoch, and unchanged original configurations.
- `provenance/`: archived code and protocols for a correction confined to the
  aggregation function. The audit verifies by AST comparison that no training
  function changed; the initial running workers used that archived source.

The report presents the complete grid and per-seed/direction details before the
initial alpha=1 and earlier projection comparisons. This is exploratory
sensitivity on an already-inspected test split, without test-based selection of
a coefficient, checkpoint or candidate. The script can resume saved fits and
supports `--setting`, `--alpha` and `--seed` for individual runs.

```sh
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_sweep -v
.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py train
.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py aggregate
.venv-cf-feasibility/bin/python cf_feasibility_sweep_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py
```

## Initial alpha=1 comparison: added feasibility penalties, no projection

The new C1/C2 runs retain their released native losses and add penalties for the
extra CounterNet feasibility rules. The six original baseline runs are reused
read-only. New runs have the same data, split, frozen classifier, initialization,
batch order, training budget and evaluation noise as their matched baseline.

For raw sigmoid output `r` and factual input `x`, the extra loss is:

```text
lambda_F * mean_draws_people(
    sum_j_in_fixed_coordinates(abs(r_j - x_j))
    + relu(r_workclass_unknown) + relu(r_occupation_unknown))
```

The fixed coordinates comprise all nine coordinates of race, gender and marital
status. Each equality is two opposing hinge inequalities. Each category ban
encourages the corresponding score to zero. The existing age hinge remains
unchanged and is not counted twice. C2's native linear education-score term is
also retained. No additional education constraint is introduced.

The added terms use each setting's published feasibility coefficient: C1=192,
C2=87. Terms are summed over coordinates and averaged over people/draws, without
dimension normalization or a new coefficient. This is our specified extension
of the soft-constraint approach, not a published extra-feature variant. More
constraints increase the possible total penalty. The initial comparison used alpha=1 only; the separate sweep above varies that
added coefficient without changing native weights or selecting test checkpoints.

No policy projection is allowed during training or evaluation: a runtime guard
raises if `AdultPolicy.project` is called. Common categorical argmax and numeric
bounds remain; there is no immutable copying, age floor, forbidden-category mask
or allowed-category renormalization. Consequently, compliance is measured rather
than assumed, including the full conditional C2 rule separately.

New artifacts are in `trained_models/cf_feasibility_penalties/`:

- `c1/` and `c2/`: protocol, identical data/classifier copies and
  `loss_only_seed{42,43,44}/` checkpoints, histories and sampled candidates.
- `summary.json` and `results.csv`: original versus loss-only results, with seed
  means, SDs, paired differences and prediction-direction cohorts. Original run
  paths and evaluation fingerprints are retained rather than copying their runs.
- `audit.json`: replay of all raw draws and independent recomputation of decoded
  metrics, frozen classifier, matched initialization and no policy projection.

`cf_feasibility_penalty_experiment.py` implements and runs this comparison;
`test_cf_feasibility_penalties.py` checks constraint signs/gradients, preservation
of the original objectives and training/evaluation without projection.
`cf_feasibility_penalty_audit.py` audits the completed artifacts. The HTML report retains this comparison after the full coefficient sweep and
also keeps the earlier projection results.

The report now follows the consolidated table in `actionability_recourse.html`:
accuracy, validity, L1 proximity and proximity/29, individual immutable violations,
their union (`viol. any`), age decreases, and the union of forbidden workclass or
occupation outputs (`Other/Unk.`). The last two are **not** included in `viol. any`.
Combined validity/compliance scores remain in the original raw evaluations but
are no longer used in the report tables.

`cf_feasibility_comparison_metrics.py` recomputes these columns from saved decoded
candidates for all 18 runs, without retraining. It exports
`counternet_style_metrics.json` and `counternet_style_metrics.csv`. Its standardized
L1/L2/L3 diagnostics use the same half-squared-error formulas as the CounterNet
report, with the Model-approx classifier's softmax probabilities. These are not
the Model-approx training losses. L4 is not applicable because the classifier is
frozen and no predictor recourse regularizer was used. The report gives the exact
native C1/C2 objectives and the added penalties separately. Three seeds and ten
draws per test observation are retained; the CounterNet table uses one delivered
counterfactual per observation. Nonzero violation rates too small to display at
one decimal are shown as `<0.1%`, rather than as zero.

## Earlier projection variants

- **Original C1:** source conditional VAE and native age penalty, with the common
  unfiltered Adult data, saved CounterNet split and bidirectional generation.
- **Extended C1:** the same model and loss, adding projection before the counterfactual
  prediction and reconstruction terms: copy race, gender and marital status;
  nondecreasing age; no generated Other/Unknown workclass or occupation.
- **Original C2:** the same VAE with the released age/education constraint term.
- **Extended C2:** the C2 objective plus exactly the same projection as Extended C1.

The native age penalty remains on the raw decoder output in both variants. The
extension normalizes allowed sigmoid scores for mutable categorical blocks during
training. At evaluation both variants receive common hard categorical decoding;
only the extension enforces the additional policy. A diagnostic also projects the
original model's existing outputs at inference, without training another model.

For C2, both native age and education terms stay on raw decoder output in both
arms. The released code differs from the paper's education-hinge description:
it uses `87 * (mean(relu(x_age-r_age)) - 0.05*mean(S(r)-S(x)))`, with a linear
education score S. Its category ranks are HS-grad/School=0,
Assoc/Bachelors/Some-college=1, Masters/Prof-school=2, Doctorate=3. The paper
instead ranks Prof-school=3. We preserve the source for training and primary
evaluation, and report a sensitivity check using the paper's ranking.

Full C2 compliance is measured on decoded categories and unrounded age: education
rank must not decrease; higher rank requires strictly higher age; equal rank
requires nondecreasing age. We count every generated candidate in each prediction
direction, whereas the source evaluator filters to candidates predicted class 1.
These choices are documented rather than presented as reproduction of published
scores. A separate joint metric requires validity, projection-policy compliance
and full C2 compliance together.

## Protocol

- All 32,561 processed rows, 29 encoded coordinates; exact saved split of
  24,420 training / 8,141 test rows. No further age/income filtering.
- One shared repository classifier, seed 42, trained for 100 epochs, then frozen.
  Source architecture is two linear layers **without ReLU**. Its learning rate is
  0.01 in code, differing from 0.001 in the paper description.
- Three matched Generator seeds: 42, 43, 44. Fifty epochs each, batch size 2,048,
  50 Monte Carlo draws during training, Adam 0.01, weight decay 0.01.
- Published Adult-Age loss settings, fixed in advance: margin 0.764, validity
  weight 29, age penalty weight 192. C2 uses published Adult-Age-Ed settings:
  margin 0.344, validity weight 76, feasibility weight 87. Thus C1/C2 comparisons
  include both the education term and different loss weights; each original/
  extended pair uses identical settings. No tuning, validation-based checkpoint
  selection or test-based selection; every model uses its final epoch.
- Ten independent samples per test row, equally weighted, no best-of selection.
  Results are separated by the frozen classifier's prediction direction.
- The source reconstruction slice `s:-1` is preserved, including omission of the
  last encoded gender coordinate. The evaluation's L1 distance uses all 29.
- Necessary common compatibility changes: device-neutral code, removal of unused
  dependencies, and skipping empty target-class hinge reductions to prevent NaNs.

The input scalers and category vocabularies are the already serialized ReLax
transforms. Their upstream fitting provenance is not established by this export.

## Run

The exporter uses the existing ReLax environment. Training uses a separate PyTorch
environment so the original CounterNet dependencies remain untouched.

```sh
.venv/bin/python -m venv .venv-cf-feasibility
.venv-cf-feasibility/bin/python -m pip install -r requirements-cf-feasibility.txt
.venv/bin/python cf_feasibility_prepare.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility -v
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_c2_experiment.py prepare
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_c2 -v
.venv-cf-feasibility/bin/python cf_feasibility_c2_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_c2_audit.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_penalties -v
.venv-cf-feasibility/bin/python cf_feasibility_penalty_experiment.py all
.venv-cf-feasibility/bin/python cf_feasibility_penalty_audit.py
.venv-cf-feasibility/bin/python -m unittest test_cf_feasibility_comparison_metrics -v
.venv-cf-feasibility/bin/python cf_feasibility_comparison_metrics.py
.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py train
.venv-cf-feasibility/bin/python cf_feasibility_penalty_sweep.py aggregate
.venv-cf-feasibility/bin/python cf_feasibility_sweep_audit.py
.venv-cf-feasibility/bin/python cf_feasibility_report.py
```

The C2 output includes the pinned `generativecf/unary-ed-const-generative-cf.py`
and `generativecf/scripts/evaluation_functions.py` in its `upstream/` directory.
For a fresh checkout without these artifacts, retrieve these two files at the
revision above before running `prepare`. The preparer reuses the C1 data and
classifier byte for byte; it never trains a different classifier for C2.

Existing completed checkpoints are reused. To conduct a fresh experiment, use a
new `--out` directory containing copies of `adult_inputs.npz`, `data_manifest.json`
and the upstream reference files; do not delete the completed experiment.

Individual commands are also available:

```sh
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py classifier
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py train --variant original --seed 42
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py train --variant extended --seed 42
.venv-cf-feasibility/bin/python cf_feasibility_experiment.py aggregate
```

## Artifacts

C1 artifacts are stored under `trained_models/cf_feasibility_baseline/`;
C2 artifacts have the same structure in `trained_models/cf_feasibility_c2/`:

- `adult_inputs.npz`: encoded data, labels and exact original row indices.
- `data_manifest.json`: features, transformation metadata and input fingerprints.
- `protocol.json`: all settings, deviations and runtime versions.
- `classifier.pt` and `classifier_results.json`: shared model and history.
- `original_seed{42,43,44}/` and `extended_seed{42,43,44}/`: Generator checkpoint,
  epoch history, training timing/checksums, evaluation JSON and sampled test CFs.
- `results.csv`: each seed, variant, evaluation mode and prediction-direction cohort.
- `summary.json`: seed means, sample SDs and paired differences.
- `audit.json`: independent recomputation of saved-candidate metrics, paired
  initialization checks and feasible class-flip availability under the classifier.
- `linear_feasibility_witnesses.npz`: diagnostic feasible feature vectors obtained
  by maximizing the target-directed margin of the repository's linear classifier.
  All 8,141 test rows have such a vector. There is no distance budget: these are
  not necessarily nearby or affordable, and are not used as Generator results.
  These witnesses cover only the six-rule policy, not full C2.
- `upstream/`, `upstream_sources.json`: inspected source snapshots and hashes for
  attribution and numerical parity checks. These are reference files, not imported
  as a training package.
- `environment.txt`, `implementation_sha256.json`: reproducibility fingerprints.

C2's `audit.json` additionally checks all C2 metrics independently, including
the source and paper ranking variants. `test_cf_feasibility_c2.py` runs the actual
inspected upstream C2 training function on a small controlled batch and compares
both its loss and gradients with the implementation, then checks the full C2 rule.

`cf-feasibility-baseline.html` is generated from these saved measurements. Main
metrics use all decoded candidates. Conditional distances are included in the
CSV but need care because the sets of successful candidates can differ.

The original native soft-output validity is also saved. It must not be confused
with validity of the decoded categorical feature vectors used for main results.
