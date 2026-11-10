"""OULAD with the CounterNet paper's Table 7 settings instead of the repo defaults.

The repo's single default configuration (learning rate 0.003, Encoder [50, 10],
Generator [50, 50]) matches Table 7 for Adult and Credit in learning rate and width, but
Table 7 gives OULAD learning rate 0.001, Encoder 127 -> 200 -> 10 and Generator
20 -> 200 -> 127. This script trains, with those settings and otherwise the same setup as
oulad_grid.py (100 epochs, batch 128, seed 42, same rules), REPEATS times each:

  base         unconstrained CounterNet (compare with the paper's Tables 2 and 3)
  feasibility  the feasibility baseline (projection onto the feasibility set)

Joint training is not reproducible run to run (unseeded batch shuffle in the trainer),
hence the repeats. Nothing under trained_models/immutability_experiment is written;
results go to trained_models/oulad_table7_test/.
"""
import pickle
from pathlib import Path

import pandas as pd

import thesis_immutability_experiment as T
import oulad_grid as O
from relax.methods.counternet import CounterNet, CounterNetConfig

OUT = Path("trained_models/oulad_table7_test")
assert T.OUT_DIR.resolve() not in OUT.resolve().parents and OUT.resolve() != T.OUT_DIR.resolve()
REPEATS = 3
TABLE7 = dict(lr=0.001, enc_sizes=[200, 10], pred_sizes=[10], exp_sizes=[200])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(1, REPEATS + 1):
        for kind in ("base", "feasibility"):
            if kind == "base":
                dm = O.load_dm([])
                net = CounterNet(CounterNetConfig(**TABLE7))
            else:
                dm = O.load_dm(O.FIXED)
                cfg = T.ReachabilityProjectionConfig(**O.proj_kwargs(), **TABLE7)
                net = CounterNet(cfg, cfnet_module=T.ReachabilityProjectionModule(cfg))
            net.train(dm, batch_size=T.BATCH_SIZE, epochs=T.EPOCHS, seed=T.TRAIN_SEED)
            d = OUT / f"{kind}_run{rep}"
            d.mkdir(exist_ok=True)
            pickle.dump(net.params, open(d / "params.pkl", "wb"))
            r = O.evaluate(net, dm)
            r.update(kind=kind, run=rep)
            rows.append(r)
            print(f"{kind} run {rep}: acc {r['acc']:.4f} val {r['validity']:.4f} "
                  f"prox {r['proximity']:.3f} (per column {r['prox_norm']:.4f}) "
                  f"viol {r['viol_any']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)


if __name__ == "__main__":
    main()
