"""Prespecified added-penalty sensitivity: alpha=.1,.3,1,3,10, C1/C2.

Only alpha * lambda_F * P_extra changes. Original and alpha=1 runs are reused
read-only. Separate protocols make interrupted runs safely resumable.
"""
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import argparse
import csv
import json

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_c2_experiment as c2
import cf_feasibility_penalty_experiment as penalty
import cf_feasibility_comparison_metrics as diagnostics

OUT = base.ROOT / 'trained_models/cf_feasibility_penalty_sweep'
ALPHAS = (.1, .3, 1., 3., 10.)
NEW_ALPHAS = (.1, .3, 3., 10.)
SEEDS = penalty.SEEDS
TRAINING_FILES = ('cf_feasibility_experiment.py', 'cf_feasibility_c2_experiment.py',
                  'cf_feasibility_penalty_experiment.py', 'cf_feasibility_penalty_sweep.py')


def tag(alpha):
    return f'alpha_{alpha:g}'


def objective(alpha):
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError('alpha must be finite and nonnegative')

    def loss(raw, mean, variance, x, target, classifier, policy, cfg, extended):
        assert not extended, 'Projection is forbidden in this sweep'
        native = c2.c2_objective if isinstance(cfg, c2.C2Config) else base.objective
        original, terms = native(raw, mean, variance, x, target, classifier, policy, cfg, False)
        extra, details = penalty.added_penalties(raw, x, policy)
        weighted = (alpha * cfg.age_weight) * extra
        total = original + weighted
        terms.update(details)
        terms.update(native_total=original, added_penalty=extra,
                     added_weighted_penalty=weighted, total=total)
        return total, terms
    return loss


def source(root, setting, alpha, seed):
    _, original, _, _ = penalty.settings(setting)
    if alpha == 0:
        return original / f'original_seed{seed}'
    if alpha == 1:
        return penalty.OUT / setting / f'loss_only_seed{seed}'
    return root / setting / tag(alpha) / f'loss_only_seed{seed}'


def prepare(root, setting, alpha):
    assert alpha in NEW_ALPHAS
    cfg, original, _, _ = penalty.settings(setting)
    out = root / setting / tag(alpha)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('adult_inputs.npz', 'data_manifest.json', 'classifier.pt', 'classifier_results.json',
                 'environment.txt', 'upstream_sources.json'):
        c2.copy_identical(original / name, out / name)
    protocol = dict(setting=setting, alpha=alpha, added_weight=alpha*cfg.age_weight,
        config=asdict(cfg), seeds=list(SEEDS), projection=False,
        objective='L_original + alpha * lambda_F * P_extra',
        original_loss='L_base + 29 H_0.764 + 192 A' if setting=='c1' else 'L_base + 76 H_0.344 + 87 A - (87 * .05) D',
        sweep_alphas=list(ALPHAS), control_alpha=0,
        baseline=str(original.relative_to(base.ROOT)),
        original_protocol=json.loads((original/'protocol.json').read_text()),
        penalty_protocol=json.loads((penalty.OUT/setting/'protocol.json').read_text()),
        design='Only the added penalty coefficient changes; native weights, classifier, data, split, seeds, initialization, batch order, training budget and evaluation noise are fixed. No early stopping or test selection.',
        interpretation='Exploratory sensitivity on the already-inspected test split; all prespecified settings reported.',
        implementation_sha256={n:penalty.fingerprint(base.ROOT/n) for n in TRAINING_FILES})
    path = out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol changed: use a fresh output directory'
    else:
        base.save_json(path, protocol)
    return out, cfg


def aggregate(root=OUT):
    rows, runs, summary, paired = [], [], {}, {}
    for setting in ('c1','c2'):
        cfg, original, _, _ = penalty.settings(setting)
        a, meta = base.load_data(original)
        x = a['x'][a['test_indices']]
        y = a['y'][a['test_indices']].reshape(-1)
        classifier = base.frozen_classifier(original, cfg)
        with torch.no_grad():
            factual = classifier(torch.from_numpy(x)).argmax(-1).numpy()
        for alpha in (0., *ALPHAS):
            for seed in SEEDS:
                path = source(root,setting,alpha,seed)
                result = json.loads((path/'evaluation.json').read_text())
                runs.append(dict(setting=setting, alpha=alpha,
                    artifact_directory=str(path.relative_to(base.ROOT)),
                    evaluation_sha256=penalty.fingerprint(path/'evaluation.json'),
                    reused=alpha in (0,1), **result))
                with np.load(path/'test_candidates.npz') as saved:
                    np.testing.assert_array_equal(saved['test_indices'],a['test_indices'])
                    for cohort, mask in [('all',np.ones(len(x),bool)),('low_to_high',factual==0),('high_to_low',factual==1)]:
                        measured = diagnostics.summarize(x,y,saved['cf_decoded'],classifier,meta['features'],mask)
                        expected = result['cohorts']['decoded'][cohort]
                        for key,old in [('validity','valid'),('proximity','proximity'),('viol_race','viol_race'),('viol_gender','viol_gender'),('viol_marital_status','viol_marital_status'),('viol_age_decrease','viol_age_decrease')]:
                            np.testing.assert_allclose(measured[key],expected[old],atol=1e-7,rtol=1e-6)
                        measured['raw_validity'] = result['native_soft'][cohort]['valid']
                        measured.update({k:v for k,v in expected.items() if k.startswith('c2_')})
                        rows.append(dict(setting=setting,alpha=alpha,added_weight=alpha*cfg.age_weight,
                                         seed=seed,cohort=cohort,**measured))
            for cohort in ('all','low_to_high','high_to_low'):
                selected = [r for r in rows if (r['setting'],r['alpha'],r['cohort'])==(setting,alpha,cohort)]
                assert len(selected)==3
                keys = [k for k in selected[0] if k not in ('setting','alpha','added_weight','seed','cohort')]
                summary[f'{setting}/{alpha:g}/{cohort}'] = {k:None if selected[0][k] is None else
                    dict(mean=float(np.mean([r[k] for r in selected])),sd=float(np.std([r[k] for r in selected],ddof=1))) for k in keys}
                if alpha:
                    control = {r['seed']:r for r in rows if (r['setting'],r['alpha'],r['cohort'])==(setting,0,cohort)}
                    paired[f'{setting}/{alpha:g}/{cohort}'] = {}
                    for key in diagnostics.METRICS:
                        if key=='L4': continue
                        values = [r[key]-control[r['seed']][key] for r in selected]
                        paired[f'{setting}/{alpha:g}/{cohort}'][key] = dict(differences=values,mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
    assert len(runs)==36 and len(rows)==108
    definitions = json.loads((penalty.OUT/'counternet_style_metrics.json').read_text())['definitions']
    base.save_json(root/'summary.json',dict(alphas=list(ALPHAS),seeds=list(SEEDS),new_fits=24,
        reused_alpha1=6,reused_original=6,summary=summary,paired=paired,rows=rows,runs=runs,
        classifier=json.loads((base.DEFAULT_OUT/'classifier_results.json').read_text()),definitions=definitions,
        implementation_sha256={n:penalty.fingerprint(base.ROOT/n) for n in TRAINING_FILES}))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    print('Sweep complete: 36 runs, 108 cohort rows; wrote summary.json and results.csv',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','aggregate'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--setting',choices=['c1','c2'])
    parser.add_argument('--alpha',type=float,choices=NEW_ALPHAS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command=='aggregate':
        aggregate(args.out);return
    for setting in ([args.setting] if args.setting else ('c1','c2')):
        for alpha in ([args.alpha] if args.alpha is not None else NEW_ALPHAS):
            out,cfg=prepare(args.out,setting,alpha)
            for seed in ([args.seed] if args.seed is not None else SEEDS):
                print(f'SWEEP {setting} alpha={alpha:g} added_weight={alpha*cfg.age_weight:g} seed={seed}',flush=True)
                with patch.object(base.AdultPolicy,'project',penalty.forbidden_projection):
                    base.train_generator(out,cfg,penalty.VARIANT,seed,objective_fn=objective(alpha))
                    if not (out/f'loss_only_seed{seed}'/'evaluation.json').exists():
                        base.evaluate_generator(out,cfg,penalty.VARIANT,seed,metrics_fn=penalty.settings(setting)[3])


if __name__=='__main__':
    main()
