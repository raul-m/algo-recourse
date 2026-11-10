# CT harness for CounterNet's split

Our additions to [CounterfactualTraining.jl](https://github.com/JuliaTrustworthyAI/CounterfactualTraining.jl),
branch `camera-ready` at `5482632`. The library (`src/`) is used unchanged; all files here belong in
its experiment harness, `paper/experiments/`.

| File | Destination | Purpose |
|---|---|---|
| `CTExperiments/src/counternet_data.jl` | `paper/experiments/CTExperiments/src/` | Dataset types (`CNAdult`, `CNCredit`, `CNCreditBalanced`, `CNOULAD`) reading the data exported by `ct_export_counternet_splits.py` |
| `configs/counternet_split/` | `paper/experiments/configs/` | Run configurations per dataset and constraint setting; `ct_counternet` (feasibility set, age only increasing) is the reported one. The nested `ct_*/mlp/...` copies are written by CT's pipeline when a configuration runs |
| `counternet_style_eval.jl` | `paper/experiments/` | Evaluates trained CT models on every test row with CounterNet's metrics; usage in the file header |
| `upstream_changes.patch` | repository root | The three edited files: `model_and_data.jl` (registers the datasets), `Project.toml`, `Manifest.toml` (IntegratedGradients from GitHub) |

## Setup

```sh
git clone -b camera-ready https://github.com/JuliaTrustworthyAI/CounterfactualTraining.jl
cd CounterfactualTraining.jl
git checkout 5482632
git apply <jax-relax>/ct_harness/upstream_changes.patch
cp -R <jax-relax>/ct_harness/CTExperiments <jax-relax>/ct_harness/configs \
      <jax-relax>/ct_harness/counternet_style_eval.jl paper/experiments/
```

The files are kept exactly as they ran, including local paths: `counternet_data.jl` reads the exported
data from `CN_DATA_DIR` (its default is a local folder), and each configuration's `save_dir` points to a
local output folder. Set both before running.
