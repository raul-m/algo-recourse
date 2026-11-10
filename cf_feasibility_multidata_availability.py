"""Separate encoded-policy availability diagnostic for the frozen linear models.

These witnesses are NOT Generator results, repairs or selected test candidates.
No distance budget or full C2 age/education condition is imposed.
"""
import json
import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_multidata as experiment


def analyze():
    torch.set_num_threads(1)
    all_results={}
    for dataset in ('credit','oulad'):
        out=experiment.OUT/dataset;a,meta=experiment.load_data(out)
        clf=experiment.frozen_classifier(out,base.Config())
        first,last=clf.predict_net
        with torch.no_grad():
            effective=last.weight@first.weight
            bias=last.weight@first.bias+last.bias
            beta=(effective[1]-effective[0]).numpy()
            offset=float(bias[1]-bias[0])
        x=a['x'][a['test_indices']]
        with torch.no_grad(): factual=clf(torch.from_numpy(x)).argmax(-1).numpy()
        # Maximize the target-directed linear margin one independent block at a time.
        direction=np.where(factual==0,1,-1)
        witness=np.empty_like(x);rules=meta['policy']
        feature_ranges={}
        for f in meta['features']:
            name,s,e=f['name'],f['start'],f['end']
            coef=beta[s:e]
            if name in rules['fixed']:
                witness[:,s:e]=x[:,s:e]
            elif f['categorical']:
                allowed=np.ones((len(x),e-s),bool)
                for category in rules['banned'].get(name,[]): allowed[:,f['categories'].index(category)]=False
                if name==rules['age'] and rules['age_kind']=='ordinal':
                    allowed &= np.arange(e-s)[None]>=x[:,s:e].argmax(-1)[:,None]
                scores=direction[:,None]*coef[None]
                best=np.where(allowed,scores,-np.inf).argmax(-1)
                witness[:,s:e]=np.eye(e-s,dtype=np.float32)[best]
            else:
                lower=x[:,s] if name==rules['age'] else np.zeros(len(x),np.float32)
                witness[:,s]=np.where(direction*coef[0]>0,1,lower)
            feature_ranges[name]=float(coef.max()-coef.min()) if f['categorical'] else float(abs(coef[0]))
        with torch.no_grad():
            predicted=clf(torch.from_numpy(witness)).argmax(-1).numpy()
            actual_logits=clf(torch.from_numpy(x)).numpy()
        affine=np.einsum('ij,j->i',x.astype(float),beta.astype(float))+offset
        assert np.isfinite(affine).all() and np.isfinite(witness).all()
        np.testing.assert_allclose(affine,actual_logits[:,1]-actual_logits[:,0],atol=1e-4,rtol=1e-5)
        checks=experiment.metrics(witness[None],x,a['y'][a['test_indices']],clf,experiment.Policy(meta),np.ones(len(x),bool))
        for key,value in checks.items():
            if key.startswith('viol_') or key.startswith('out_disallowed_'): assert value==0,(dataset,key,value)
        found=predicted!=factual
        result=dict(n_test=len(x),n_with_encoded_policy_flip=int(found.sum()),fraction=float(found.mean()),
            encoded_policy_checked=True,
            label='Availability witness only; not a learned-Generator result',
            scope='Independent encoded CounterNet policy, including age nondecrease and forbidden categories; no proximity budget, affordability, causal intervention model or full C2 coupling.',
            method='The classifier is affine. For each target direction, select the maximizing numeric endpoint or permitted one-hot category in each free block, and retain fixed blocks.',
            feature_logit_ranges=feature_ranges,
            fixed_feature_logit_range_bound=sum(feature_ranges[n] for n in rules['fixed']),
            classifier_sha256=base.state_hash(clf),data_sha256=meta['encoded_x_sha256'])
        base.save_json(out/'availability.json',result)
        np.savez_compressed(out/'availability_witnesses.npz',test_indices=a['test_indices'],factual_prediction=factual,witness=witness,prediction=predicted,found=found)
        all_results[dataset]=result
    base.save_json(experiment.OUT/'availability.json',all_results)
    print(json.dumps({d:{k:v for k,v in r.items() if k not in ('feature_logit_ranges','method','scope')} for d,r in all_results.items()},indent=2))


if __name__=='__main__': analyze()
