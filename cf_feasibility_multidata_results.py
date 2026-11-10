"""Aggregate and independently audit the Credit/OULAD Model-approx experiments."""
import argparse
import csv
from dataclasses import asdict
import json

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_multidata as experiment


def aggregate(root=experiment.OUT):
    rows,runs,summary,paired=[],[],{},{}
    manifests={};classifiers={}
    for dataset in ('credit','oulad'):
        out=root/dataset
        manifests[dataset]=json.loads((out/'data_manifest.json').read_text())
        classifiers[dataset]=json.loads((out/'classifier_results.json').read_text())
        for setting in ('c1','c2'):
            cfg=experiment.configuration(setting)
            for alpha in experiment.ALPHAS:
                for seed in experiment.SEEDS:
                    path=experiment.run_path(out,alpha,seed,setting)
                    result=json.loads((path/'evaluation.json').read_text())
                    assert (result['dataset'],result['setting'],result['alpha'],result['seed'])==(dataset,setting,alpha,seed)
                    history=json.loads((path/'history.json').read_text())
                    runs.append(dict(**result,artifact_directory=str(path.relative_to(base.ROOT)),
                        evaluation_sha256=experiment.fingerprint(path/'evaluation.json'),
                        final_training_terms=history[-1],training_seconds=json.loads((path/'training.json').read_text())['seconds']))
                    for cohort,values in result['cohorts'].items():
                        rows.append(dict(dataset=dataset,setting=setting,alpha=alpha,added_weight=alpha*cfg.age_weight,
                            seed=seed,cohort=cohort,**values))
                for cohort in ('all','0_to_1','1_to_0'):
                    selected=[r for r in rows if (r['dataset'],r['setting'],r['alpha'],r['cohort'])==(dataset,setting,alpha,cohort)]
                    assert {r['seed'] for r in selected}==set(experiment.SEEDS)
                    keys=[k for k in selected[0] if k not in ('dataset','setting','alpha','added_weight','seed','cohort')]
                    summary[f'{dataset}/{setting}/{alpha:g}/{cohort}']={k:None if selected[0][k] is None else
                        dict(mean=float(np.mean([r[k] for r in selected])),sd=float(np.std([r[k] for r in selected],ddof=1))) for k in keys}
                    if alpha and selected[0]['n_test']:
                        baseline={r['seed']:r for r in rows if (r['dataset'],r['setting'],r['alpha'],r['cohort'])==(dataset,setting,0,cohort)}
                        diffs={k:[r[k]-baseline[r['seed']][k] for r in selected] for k in keys if k not in ('L4','n_test','draws_per_person')}
                        paired[f'{dataset}/{setting}/{alpha:g}/{cohort}']={k:dict(differences=v,mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) for k,v in diffs.items()}
    assert len(runs)==72 and len(rows)==216
    result=dict(runs=runs,rows=rows,summary=summary,paired=paired,manifests=manifests,classifiers=classifiers,
        alphas=list(experiment.ALPHAS),seeds=list(experiment.SEEDS),
        protocols={d:json.loads((root/d/'protocol.json').read_text()) for d in ('credit','oulad')},
        interpretation='Transferred Adult C1/C2 weights and raw-range reconstruction, with explicit dataset-specific age/education conversions. No test-based selection or retuning.')
    base.save_json(root/'summary.json',result)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    print('Aggregated 72 runs and 216 cohort rows.',flush=True)


def independent_metrics(cf,raw,x,y,clf,features,policy,mask):
    """Use independently decoded category indices and NumPy arithmetic."""
    x=x[mask];y=y[mask];cf=cf[:,mask];raw=raw[:,mask]
    with torch.no_grad():
        factual_logits=clf(torch.from_numpy(x)).numpy().astype(float)
        cf_logits=clf(torch.from_numpy(cf)).numpy().astype(float)
        raw_logits=clf(torch.from_numpy(raw)).numpy()
    factual=factual_logits.argmax(-1);pred=cf_logits.argmax(-1)
    def prob(z):
        z=np.exp(z-z.max(-1,keepdims=True));return z[...,1]/z.sum(-1)
    p=prob(factual_logits);q=prob(cf_logits)
    distance=np.abs(cf-x[None]).sum(-1)
    metrics=dict(n_test=len(x),draws_per_person=cf.shape[0],acc=(factual==y).mean(),validity=(pred!=factual[None]).mean(),
        L1=.5*np.square(p-y).mean(),L2=.5*np.square(q-(1-factual[None])).mean(),
        L3=.5*np.square(cf.astype(float)-x[None]).mean(),L4=None,proximity=distance.mean(),prox_norm=distance.mean()/x.shape[-1],
        raw_validity=(raw_logits.argmax(-1)!=factual[None]).mean())
    by_name={f['name']:f for f in features}
    flags=[]
    for name in policy['fixed']:
        f=by_name[name];s,e=f['start'],f['end']
        flag=cf[...,s:e].argmax(-1)!=x[None,:,s:e].argmax(-1)
        flags.append(flag);metrics['viol_'+name]=flag.mean()
    metrics['viol_any']=np.logical_or.reduce(flags).mean()
    f=by_name[policy['age']];s,e=f['start'],f['end']
    age=(cf[...,s]-x[None,:,s]<-1e-6) if policy['age_kind']=='continuous' else (cf[...,s:e].argmax(-1)<x[None,:,s:e].argmax(-1))
    metrics['viol_age_decrease']=age.mean()
    flags=[]
    for name,categories in policy['banned'].items():
        f=by_name[name];s,e=f['start'],f['end']
        forbidden=[f['categories'].index(c) for c in categories]
        flag=np.isin(cf[...,s:e].argmax(-1),forbidden)
        metrics['out_disallowed_'+name]=flag.mean();flags.append(flag)
    metrics['out_disallowed_any']=np.logical_or.reduce(flags).mean()
    return metrics


def audit(root=experiment.OUT):
    summary=json.loads((root/'summary.json').read_text())
    details=[];checked=0
    for dataset in ('credit','oulad'):
        out=root/dataset;a,meta=experiment.load_data(out)
        protocol=summary['protocols'][dataset]
        for name,digest in protocol['implementation_sha256'].items(): assert experiment.fingerprint(base.ROOT/name)==digest
        for name,digest in meta['source_sha256'].items(): assert experiment.fingerprint(base.ROOT/meta['source_directory']/name)==digest
        assert experiment.fingerprint(base.ROOT/meta['policy_source'])==meta['policy_source_sha256']
        np.testing.assert_array_equal(np.sort(np.r_[a['train_indices'],a['test_indices']]),np.arange(len(a['x'])))
        assert len(np.intersect1d(a['train_indices'],a['test_indices']))==0
        x=a['x'][a['test_indices']];y=a['y'][a['test_indices']]
        classifier=experiment.frozen_classifier(out,base.Config());clf_hash=base.state_hash(classifier)
        assert clf_hash==summary['classifiers'][dataset]['state_sha256']
        with torch.no_grad(): factual=classifier(torch.from_numpy(x)).argmax(-1).numpy()
        np.testing.assert_allclose((factual==y).mean(),summary['classifiers'][dataset]['test']['accuracy'],rtol=0,atol=1e-12)
        initial={}
        for run in [r for r in summary['runs'] if r['dataset']==dataset]:
            setting,alpha,seed=run['setting'],run['alpha'],run['seed']
            cfg=experiment.configuration(setting)
            path=base.ROOT/run['artifact_directory']
            assert path==experiment.run_path(out,alpha,seed,setting)
            assert experiment.fingerprint(path/'evaluation.json')==run['evaluation_sha256']
            saved=torch.load(path/'generator.pt',map_location='cpu',weights_only=True)
            assert saved['config']==asdict(cfg) and saved['classifier_sha256']==clf_hash
            assert saved['dataset']==dataset and saved['setting']==setting and saved['alpha']==alpha and saved['seed']==seed
            assert saved['data_sha256']==meta['encoded_x_sha256'] and saved['dimensions']==meta['dimensions']
            base.seed_all(seed);fresh=base.ModelApproxCF(meta['dimensions'],cfg.latent_dim)
            assert base.state_hash(fresh)==saved['initial_sha256']
            initial.setdefault(seed,saved['initial_sha256']);assert initial[seed]==saved['initial_sha256']
            history=json.loads((path/'history.json').read_text())
            assert [r['epoch'] for r in history]==list(range(1,cfg.epochs+1))
            for row in history:
                assert all(np.isfinite(v) for v in row.values())
                np.testing.assert_allclose(row['added_weighted_penalty'],alpha*cfg.age_weight*row['added_penalty'],rtol=2e-6)
                np.testing.assert_allclose(row['total'],row['native_total']+row['added_weighted_penalty'],rtol=2e-6)
                expected_native=row['reconstruction']+row['simplex']+row['kl']+cfg.validity_weight*row['validity_hinge']+cfg.age_weight*(row['age_penalty']+row['education_term'])
                np.testing.assert_allclose(row['native_total'],expected_native,rtol=2e-6)
            model=base.ModelApproxCF(meta['dimensions'],cfg.latent_dim).eval();model.load_state_dict(saved['state_dict'])
            assert base.state_hash(model)==json.loads((path/'training.json').read_text())['final_sha256']
            with np.load(path/'test_candidates.npz') as arrays:
                raw=arrays['cf_native_soft'];cf=arrays['cf_decoded']
                assert raw.shape==cf.shape==(cfg.eval_samples,len(x),meta['dimensions'])
                assert np.isfinite(raw).all() and np.isfinite(cf).all()
                np.testing.assert_array_equal(arrays['test_indices'],a['test_indices'])
                np.testing.assert_array_equal(arrays['factual_prediction'],factual)
                decoded=np.empty_like(raw)
                for f in meta['features']:
                    s,e=f['start'],f['end']
                    decoded[...,s:e]=np.eye(e-s,dtype=np.float32)[raw[...,s:e].argmax(-1)] if f['categorical'] else raw[...,s:e].clip(0,1)
                np.testing.assert_array_equal(decoded,cf)
                rng=torch.Generator().manual_seed(cfg.eval_seed+seed)
                with torch.no_grad():
                    for offset in range(0,len(x),cfg.eval_batch):
                        batch=torch.from_numpy(x[offset:offset+cfg.eval_batch]);target=1-classifier(batch).argmax(-1)
                        mean,var=model.encode(batch,target)
                        z=mean[None]+var.sqrt()[None]*torch.randn(cfg.eval_samples,len(batch),cfg.latent_dim,generator=rng)
                        label=target.float()[None,:,None].expand(cfg.eval_samples,-1,-1)
                        replay=model.decoder_mean(torch.cat([z,label],-1).reshape(-1,cfg.latent_dim+1)).reshape(cfg.eval_samples,len(batch),meta['dimensions'])
                        np.testing.assert_array_equal(replay.numpy(),raw[:,offset:offset+len(batch)])
                    np.testing.assert_array_equal(arrays['prediction_decoded'],classifier(torch.from_numpy(cf)).argmax(-1).numpy())
                    np.testing.assert_array_equal(arrays['prediction_native_soft'],classifier(torch.from_numpy(raw)).argmax(-1).numpy())
                for cohort,expected in run['cohorts'].items():
                    mask=np.ones(len(x),bool) if cohort=='all' else factual==(0 if cohort=='0_to_1' else 1)
                    if not mask.any(): assert expected['n_test']==0;continue
                    measured=independent_metrics(cf,raw,x,y,classifier,meta['features'],meta['policy'],mask)
                    assert measured.keys()==expected.keys()
                    for key,value in measured.items():
                        if value is None: assert expected[key] is None
                        else: np.testing.assert_allclose(value,expected[key],rtol=2e-6,atol=1e-8,err_msg=f'{dataset} {setting} {alpha} {seed} {cohort} {key}')
                checked+=cf.shape[0]*cf.shape[1]
            details.append(dict(dataset=dataset,setting=setting,alpha=alpha,seed=seed,artifact_directory=run['artifact_directory'],
                checkpoint_sha256=experiment.fingerprint(path/'generator.pt'),candidates_sha256=experiment.fingerprint(path/'test_candidates.npz'),
                raw_replay_exact=True,ordinary_decoding_exact=True))
            print(f'Audited {dataset} {setting} alpha={alpha:g} seed={seed}',flush=True)
    assert len(details)==72 and checked==5633640
    assert len({(r['dataset'],r['setting'],r['alpha'],r['seed']) for r in details})==72
    for key,values in summary['summary'].items():
        dataset,setting,alpha,cohort=key.split('/')
        selected=[r for r in summary['rows'] if (r['dataset'],r['setting'],r['alpha'],r['cohort'])==(dataset,setting,float(alpha),cohort)]
        assert len(selected)==3
        for metric,expected in values.items():
            if expected is None: assert all(r[metric] is None for r in selected)
            else:
                samples=[r[metric] for r in selected]
                np.testing.assert_allclose(expected['mean'],np.mean(samples),rtol=0,atol=1e-12)
                np.testing.assert_allclose(expected['sd'],np.std(samples,ddof=1),rtol=0,atol=1e-12)
    result=dict(passed=True,runs=len(details),decoded_candidates_audited=checked,raw_draws_replayed=checked,
        checks=['Unchanged saved data source files, split indices, encodings and policy sources',
            'Exact configurations, native/added coefficient arithmetic and complete training histories',
            'Same Generator initialization per seed across both settings and every alpha',
            'Frozen classifier and opposite-label targets, both directions, ten draws per row',
            'Exact replay of every raw sample and independent ordinary decoding with no policy projection',
            'Independent NumPy diagnostics and policy violations; all means/SDs checked'],
        details=details,implementation_sha256=experiment.fingerprint(base.ROOT/'cf_feasibility_multidata_results.py'))
    base.save_json(root/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['aggregate','audit','all'])
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.command in ('aggregate','all'): aggregate()
    if args.command in ('audit','all'): audit()
