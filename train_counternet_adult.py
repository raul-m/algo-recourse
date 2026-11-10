"""Train CounterNet on the Adult income dataset and evaluate the CF explanations.

Reports metrics on two populations:
  - full:  all 32,561 instances (train + test), the protocol used by
           benchmarks/built-in/run_all.py and relax.generate_cf_explanations
  - test:  the held-out test split only (8,141 instances, 25%), matching the
           evaluation protocol of the CounterNet paper

If a saved model exists at trained_models/counternet_adult/params.pkl, it is
loaded instead of retraining (delete the file or pass --retrain to retrain).
"""
import argparse
import pickle
import time
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

import relax
from relax.methods import CounterNet
from relax.explain import generate_cf_explanations
from relax.evaluate import compute_validity, compute_proximity

EPOCHS = 100
BATCH_SIZE = 128
OUT_DIR = Path("trained_models/counternet_adult")
PARAMS_PATH = OUT_DIR / "params.pkl"
N_FEATURES = 29  # Adult input dim after one-hot; used to normalize proximity


def load_trained(cfnet: CounterNet, dm) -> CounterNet:
    cfnet.set_data_module(dm)
    cfnet.module = cfnet._init_model(cfnet.config, cfnet.module)
    cfnet.module._data_module = dm
    with open(PARAMS_PATH, "rb") as f:
        cfnet.params = pickle.load(f)
    cfnet._is_trained = True
    return cfnet


def evaluate_population(cfnet: CounterNet, xs, ys) -> dict:
    cfs = jnp.stack([cfnet.generate_cf(x) for x in xs])
    preds = cfnet.pred_fn(xs)
    acc = float((jnp.argmax(preds, axis=-1) == ys.reshape(-1)).mean())
    validity = float(compute_validity(xs, cfs, cfnet.pred_fn))
    proximity = float(compute_proximity(xs, cfs))
    return {
        "n": xs.shape[0],
        "acc": acc,
        "validity": validity,
        "proximity": proximity,
        "proximity_normalized": proximity / N_FEATURES,
    }


def main(retrain: bool):
    print("Loading Adult dataset...")
    dm = relax.load_data("adult")
    train_xs, train_ys = dm["train"]
    test_xs, test_ys = dm["test"]
    print(f"train: {train_xs.shape}, test: {test_xs.shape}")

    cfnet = CounterNet()
    if PARAMS_PATH.exists() and not retrain:
        print(f"Loading saved model from {PARAMS_PATH} (pass --retrain to retrain)")
        cfnet = load_trained(cfnet, dm)
    else:
        print(f"Training CounterNet for {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
        start = time.time()
        cfnet.train(dm, batch_size=BATCH_SIZE, epochs=EPOCHS)
        print(f"Training finished in {time.time() - start:.1f}s")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_PATH, "wb") as f:
            pickle.dump(cfnet.params, f)
        print(f"Saved trained params to {PARAMS_PATH}")
        # hook up constraints for generate_cf/evaluation
        cfnet.set_data_module(dm)

    print("Evaluating...")
    results = {
        "full (train+test)": evaluate_population(cfnet, dm.xs, dm.ys),
        "test only": evaluate_population(cfnet, test_xs, test_ys),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "population"
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "metrics.csv")
    print(f"Saved metrics to {OUT_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="retrain even if a saved model exists")
    args = parser.parse_args()
    main(args.retrain)
