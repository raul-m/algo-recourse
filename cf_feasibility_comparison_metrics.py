"""CounterNet-style test diagnostics from completed Model-approx checkpoints.

No training or candidate generation: read saved decoded candidates. Keep these
reporting diagnostics separate from the native Model-approx training losses.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_c2_experiment as c2

OUT = base.ROOT / 'trained_models/cf_feasibility_penalties'
METRICS = ('acc','validity','L1','L2','L3','L4','proximity','prox_norm',
           'viol_race','viol_gender','viol_marital_status','viol_any',
           'viol_age_decrease','out_disallowed_any')
VARIANTS = ('original','loss_only','projected')


def summarize(x, y, cf, classifier, features, mask):
    x, y, cf = x[mask], y[mask], cf[:,mask]
    with torch.no_grad():
        factual_logits = classifier(torch.from_numpy(x))
        cf_logits = classifier(torch.from_numpy(cf))
        factual = factual_logits.argmax(-1).numpy()
        prediction = cf_logits.argmax(-1).numpy()
        p = factual_logits.softmax(-1)[:,1].numpy()
        q = cf_logits.softmax(-1)[...,1].numpy()
    distance = np.abs(cf-x[None]).sum(-1)
    values = dict(acc=float((factual==y).mean()),validity=float((prediction!=factual[None]).mean()),
        L1=float(.5*np.square(p.astype(float)-y).mean()),
        L2=float(.5*np.square(q.astype(float)-(1-factual[None])).mean()),
        L3=float(.5*np.square(cf.astype(float)-x[None]).mean()),L4=None,
        proximity=float(distance.mean()),prox_norm=float(distance.mean()/x.shape[-1]))
    by_name = {f['name']:f for f in features}
    changed = []
    for name in ('race','gender','marital_status'):
        f=by_name[name];s,e=f['start'],f['end']
        violation = np.abs(cf[...,s:e]-x[None,:,s:e]).sum(-1)>1e-6
        # Independent category-index check on these delivered one-hot vectors.
        np.testing.assert_array_equal(violation,cf[...,s:e].argmax(-1)!=x[None,:,s:e].argmax(-1))
        values['viol_'+name]=float(violation.mean());changed.append(violation)
    values['viol_any']=float(np.logical_or.reduce(changed).mean())
    assert max(values['viol_'+n] for n in ('race','gender','marital_status')) <= values['viol_any']+1e-12
    assert values['viol_any'] <= min(1,sum(values['viol_'+n] for n in ('race','gender','marital_status')))+1e-12
    age=by_name['age']['start']
    values['viol_age_decrease']=float((cf[...,age]<x[None,:,age]-1e-6).mean())
    unknown=[]
    for name in ('workclass','occupation'):
        f=by_name[name];j=f['start']+f['categories'].index('Other/Unknown')
        flag=cf[...,j]>.5;unknown.append(flag)
        values['out_'+name+'_other_unknown']=float(flag.mean())
    values['out_disallowed_any']=float(np.logical_or.reduce(unknown).mean())
    values.update(n_test=len(x),draws_per_person=cf.shape[0])
    return values


def build():
    torch.set_num_threads(1)
    rows, sources = [], []
    for setting,old,cfg in [('c1',base.DEFAULT_OUT,base.Config()),('c2',c2.OUT,c2.C2Config())]:
        a,meta=base.load_data(old)
        x=a['x'][a['test_indices']];y=a['y'][a['test_indices']].reshape(-1)
        classifier=base.frozen_classifier(old,cfg)
        with torch.no_grad(): factual=classifier(torch.from_numpy(x)).argmax(-1).numpy()
        for variant in VARIANTS:
            directory=OUT/setting if variant=='loss_only' else old
            artifact_variant='extended' if variant=='projected' else variant
            for seed in (42,43,44):
                path=directory/f'{artifact_variant}_seed{seed}'
                evaluation=json.loads((path/'evaluation.json').read_text())
                with np.load(path/'test_candidates.npz') as saved:
                    np.testing.assert_array_equal(saved['test_indices'],a['test_indices'])
                    cf=saved['cf_decoded']
                    for cohort,mask in [('all',np.ones(len(x),bool)),('low_to_high',factual==0),('high_to_low',factual==1)]:
                        measured=summarize(x,y,cf,classifier,meta['features'],mask)
                        expected=evaluation['cohorts']['decoded'][cohort]
                        for k,old_key in [('validity','valid'),('proximity','proximity'),('viol_race','viol_race'),('viol_gender','viol_gender'),('viol_marital_status','viol_marital_status'),('viol_age_decrease','viol_age_decrease'),('out_workclass_other_unknown','viol_workclass_unknown'),('out_occupation_other_unknown','viol_occupation_unknown')]:
                            np.testing.assert_allclose(measured[k],expected[old_key],atol=1e-7,rtol=1e-6)
                        rows.append(dict(setting=setting,variant=variant,seed=seed,cohort=cohort,**measured))
                sources.append(dict(setting=setting,variant=variant,seed=seed,
                    candidates=str((path/'test_candidates.npz').relative_to(base.ROOT)),
                    sha256=hashlib.sha256((path/'test_candidates.npz').read_bytes()).hexdigest()))
    summary={}
    for setting in ('c1','c2'):
        for variant in VARIANTS:
            for cohort in ('all','low_to_high','high_to_low'):
                selected=[r for r in rows if (r['setting'],r['variant'],r['cohort'])==(setting,variant,cohort)]
                assert len(selected)==3
                summary[f'{setting}/{variant}/{cohort}']={k:None if selected[0][k] is None else
                    dict(mean=float(np.mean([r[k] for r in selected])),sd=float(np.std([r[k] for r in selected],ddof=1)))
                    for k in (*METRICS,'n_test','draws_per_person')}
    definitions=dict(
        validity='Fraction of all saved decoded candidates that flip the original predicted label; both directions.',
        proximity='Mean sum of absolute differences across all 29 encoded columns; prox_norm divides by 29.',
        viol_any='Union of changes to race, gender and marital_status ONLY; not all six feasibility rules.',
        out_disallowed_any='Union of generated Other/Unknown workclass or occupation; not their summed rates and not Other race.',
        L1='0.5 * mean((softmax(f(x))[1] - y)^2), a standardized prediction-error diagnostic.',
        L2='0.5 * mean((softmax(f(cf))[1] - (1-argmax(f(x))))^2), a standardized validity-error diagnostic.',
        L3='0.5 * mean((cf-x)^2) over draws, observations and all 29 columns; not L1 proximity.',
        L4='Not applicable: no predictor-side recourse regularizer in these post-hoc Model-approx runs.',
        loss_scope='L1-L3 use the CounterNet report formulas as test diagnostics. They are NOT the Model-approx reconstruction/KL/hinge training objective. Probabilities are the frozen two-logit classifier softmax.',
        averaging='Three seed means; sample SD across Generator seeds. Each seed counts all 10 draws for each of 8141 test rows, unlike the single delivered CF per row in the CounterNet table.',
        tolerance='Immutable block L1 difference >1e-6; age < factual age -1e-6; forbidden category coordinate >0.5.')
    result=dict(definitions=definitions,summary=summary,rows=rows,sources=sources,
        checks=dict(passed=True,runs=18,decoded_candidates=18*8141*10,
                    original_metrics_match=True,immutable_union_checked=True),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    base.save_json(OUT/'counternet_style_metrics.json',result)
    with (OUT/'counternet_style_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote CounterNet-style metrics: 18 runs, 54 cohort rows; existing metrics match.')


if __name__=='__main__':
    build()
