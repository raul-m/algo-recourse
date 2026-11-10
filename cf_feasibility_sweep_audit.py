"""Audit all 36 sweep/control runs: exact replay, losses, decoding and metrics."""
from dataclasses import asdict
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_penalty_experiment as experiment
import cf_feasibility_penalty_sweep as sweep


def independent_metrics(cf, x, pred, factual, features, c2=False, paper=False):
    by_name = {f['name']:f for f in features}
    result = dict(valid=pred != factual[None], proximity=np.abs(cf-x[None]).sum(-1))
    for name in ('race','gender','marital_status'):
        f = by_name[name]
        result['viol_'+name] = np.any(cf[...,f['start']:f['end']] != x[None,:,f['start']:f['end']],axis=-1)
    age = by_name['age']['start']
    result['viol_age_decrease'] = cf[...,age] < x[None,:,age]-1e-6
    for name in ('workclass','occupation'):
        f = by_name[name]
        index = f['start']+f['categories'].index('Other/Unknown')
        result['viol_'+name+'_unknown'] = cf[...,index] == 1
    result['feasible'] = ~np.logical_or.reduce([v for k,v in result.items() if k.startswith('viol_')])
    result['valid_feasible'] = result['valid'] & result['feasible']
    if c2:
        f = by_name['education']
        rank_map = {'HS-grad':0,'School':0,'Bachelors':1,'Assoc':1,'Some-college':1,'Masters':2,'Prof-school':2,'Doctorate':3}
        for suffix, professional_rank in [('',2),('_paper_rank',3)]:
            ranks = np.array([{**rank_map,'Prof-school':professional_rank}[v] for v in f['categories']])
            delta = ranks[cf[...,f['start']:f['end']].argmax(-1)] - ranks[x[:,f['start']:f['end']].argmax(-1)][None]
            age_delta = cf[...,age]-x[None,:,age]
            # Express as the complement of the three disjoint failure cases.
            failure = (delta<0) | ((delta>0)&(age_delta<=0)) | ((delta==0)&(age_delta<0))
            result['c2_compliant'+suffix] = ~failure
            result['valid_c2'+suffix] = result['valid'] & ~failure
            result['feasible_c2'+suffix] = result['feasible'] & ~failure
            result['valid_feasible_c2'+suffix] = result['valid_feasible'] & ~failure
            if not suffix:
                result['c2_education_decrease'] = delta<0
                result['c2_higher_education_without_older_age'] = (delta>0)&(age_delta<=0)
                result['c2_same_education_with_younger_age'] = (delta==0)&(age_delta<0)
    return result


def audit(root=sweep.OUT):
    torch.set_num_threads(1)
    summary = json.loads((root/'summary.json').read_text())
    checked, details = 0, []
    reused = json.loads((experiment.OUT/'audit.json').read_text())
    assert reused['passed']
    reused_hashes = {r['artifact_directory']:r for r in reused['details']}
    # A reporting-only correction was made while training was in progress.
    # Retain the exact prior source and protocols, and prove training unchanged.
    amendment_path=root/'provenance/aggregation_fix.json'
    if amendment_path.exists():
        amendment=json.loads(amendment_path.read_text())
        archived=root/'provenance/cf_feasibility_penalty_sweep_before_aggregation_fix.py'
        current=base.ROOT/'cf_feasibility_penalty_sweep.py'
        assert experiment.fingerprint(archived)==amendment['old_sha256']
        assert experiment.fingerprint(current)==amendment['new_sha256']
        definitions=lambda p:{n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
        old_defs,new_defs=definitions(archived),definitions(current)
        assert old_defs.keys()==new_defs.keys()
        assert [k for k in old_defs if old_defs[k]!=new_defs[k]]==['aggregate']
    for name, digest in summary['implementation_sha256'].items():
        assert experiment.fingerprint(base.ROOT/name)==digest
    for setting in ('c1','c2'):
        old_protocol = json.loads((experiment.OUT/setting/'protocol.json').read_text())
        for name,digest in old_protocol['implementation_sha256'].items():
            assert experiment.fingerprint(base.ROOT/name)==digest
        for alpha in sweep.NEW_ALPHAS:
            sweep.prepare(root,setting,alpha)  # refuses protocol/hash changes

    for run in summary['runs']:
        setting, variant, seed = run['setting'],run['variant'],run['seed']
        cfg, baseline, _, _ = experiment.settings(setting)
        path = base.ROOT/run['artifact_directory']
        alpha = run['alpha']
        assert path == sweep.source(root,setting,alpha,seed)
        out = path.parent
        if alpha in (0,1):
            previous=reused_hashes[run['artifact_directory']]
            assert experiment.fingerprint(path/'generator.pt')==previous['generator_file_sha256']
            assert experiment.fingerprint(path/'test_candidates.npz')==previous['candidates_file_sha256']

        for name in ('adult_inputs.npz','classifier.pt','data_manifest.json'):
            assert experiment.fingerprint(out/name) == experiment.fingerprint(baseline/name)
        a, meta = base.load_data(out)
        x = a['x'][a['test_indices']]
        classifier = base.frozen_classifier(out,cfg)
        with torch.no_grad():
            factual = classifier(torch.from_numpy(x)).argmax(-1).numpy()
        accuracy = float((factual==a['y'][a['test_indices']].reshape(-1)).mean())
        np.testing.assert_allclose(accuracy,summary['classifier']['test']['accuracy'],atol=1e-12)
        path = base.ROOT/run['artifact_directory']
        assert experiment.fingerprint(path/'evaluation.json') == run['evaluation_sha256']
        trained = torch.load(path/'generator.pt',map_location='cpu',weights_only=True)
        original = torch.load(baseline/f'original_seed{seed}/generator.pt',map_location='cpu',weights_only=True)
        assert trained['config'] == asdict(cfg) == original['config']
        assert trained['initial_sha256'] == original['initial_sha256']
        assert trained['classifier_sha256'] == original['classifier_sha256'] == base.state_hash(classifier)
        assert trained['data_sha256'] == original['data_sha256'] == meta['encoded_x_sha256']
        history=json.loads((path/'history.json').read_text())
        assert len(history)==cfg.epochs
        assert trained['seed']==seed and trained['variant']==variant
        if alpha:
            for row in history:
                np.testing.assert_allclose(row['added_weighted_penalty'],alpha*cfg.age_weight*row['added_penalty'],rtol=2e-6)
                np.testing.assert_allclose(row['total'],row['native_total']+row['added_weighted_penalty'],rtol=2e-6)
                detail_sum=sum(row[k] for k in ('penalty_race','penalty_gender','penalty_marital_status','penalty_workclass_unknown','penalty_occupation_unknown'))
                np.testing.assert_allclose(row['added_penalty'],detail_sum,rtol=2e-6)

        model = base.ModelApproxCF(latent_dim=cfg.latent_dim).eval()
        model.load_state_dict(trained['state_dict'])
        assert base.state_hash(model) == json.loads((path/'training.json').read_text())['final_sha256']
        with np.load(path/'test_candidates.npz') as saved:
            np.testing.assert_array_equal(saved['test_indices'],a['test_indices'])
            np.testing.assert_array_equal(saved['factual_prediction'],factual)
            cf, raw = saved['cf_decoded'], saved['cf_native_soft']
            assert cf.shape == raw.shape == (cfg.eval_samples,len(x),29)
            assert np.isfinite(cf).all() and np.isfinite(raw).all()
            # Reconstruct the common representation cleanup independently.
            decoded = np.empty_like(raw)
            for f in meta['features']:
                s,e = f['start'],f['end']
                if f['categorical']:
                    decoded[...,s:e] = np.eye(e-s,dtype=np.float32)[raw[...,s:e].argmax(-1)]
                else:
                    decoded[...,s:e] = raw[...,s:e].clip(0,1)
            np.testing.assert_array_equal(cf,decoded)
            # Replay every raw draw from the frozen Generator and matched noise.
            rng = torch.Generator().manual_seed(cfg.eval_seed+seed)
            with torch.no_grad():
                for offset in range(0,len(x),cfg.eval_batch):
                    batch = torch.from_numpy(x[offset:offset+cfg.eval_batch])
                    target = 1-classifier(batch).argmax(-1)
                    mean,var = model.encode(batch,target)
                    noise = torch.randn(cfg.eval_samples,len(batch),cfg.latent_dim,generator=rng)
                    z = mean[None]+var.sqrt()[None]*noise
                    labels = target.float()[None,:,None].expand(cfg.eval_samples,-1,-1)
                    replay = model.decoder_mean(torch.cat([z,labels],-1).reshape(-1,cfg.latent_dim+1)).reshape(cfg.eval_samples,len(batch),29)
                    np.testing.assert_array_equal(replay.numpy(),raw[:,offset:offset+len(batch)])
                prediction = classifier(torch.from_numpy(cf)).argmax(-1).numpy()
                soft_prediction = classifier(torch.from_numpy(raw)).argmax(-1).numpy()
            np.testing.assert_array_equal(prediction,saved['prediction_decoded'])
            np.testing.assert_array_equal(soft_prediction,saved['prediction_native_soft'])
            metrics = independent_metrics(cf,x,prediction,factual,meta['features'],c2=setting=='c2')
            for cohort, expected in run['cohorts']['decoded'].items():
                mask = np.ones(len(x),bool) if cohort=='all' else factual==(0 if cohort=='low_to_high' else 1)
                for name, value in metrics.items():
                    np.testing.assert_allclose(value[:,mask].mean(),expected[name],rtol=1e-6,atol=1e-8)
                for name in ('valid','valid_feasible'):
                    selected = metrics[name][:,mask]
                    value = metrics['proximity'][:,mask][selected].mean() if selected.any() else None
                    if value is None: assert expected['proximity_'+name] is None
                    else: np.testing.assert_allclose(value,expected['proximity_'+name],rtol=1e-6)
                soft = run['native_soft'][cohort]
                np.testing.assert_allclose((soft_prediction[:,mask]!=factual[None,mask]).mean(),soft['valid'])
                np.testing.assert_allclose(np.abs(raw[:,mask]-x[None,mask]).sum(-1).mean(),soft['proximity'])
            # Independently verify report diagnostics, using NumPy probabilities.
            with torch.no_grad():
                logits=classifier(torch.from_numpy(x)).numpy().astype(float)
                cf_logits=classifier(torch.from_numpy(cf)).numpy().astype(float)
            def probability(logits):
                stable=np.exp(logits-logits.max(axis=-1,keepdims=True))
                return stable[...,1]/stable.sum(-1)
            p,q=probability(logits),probability(cf_logits)
            y=a['y'][a['test_indices']].reshape(-1)
            for row in summary['rows']:
                if (row['setting'],row['alpha'],row['seed'])!=(setting,alpha,seed): continue
                cohort=row['cohort']
                mask=np.ones(len(x),bool) if cohort=='all' else factual==(0 if cohort=='low_to_high' else 1)
                expected=dict(acc=(factual[mask]==y[mask]).mean(),
                    validity=(prediction[:,mask]!=factual[None,mask]).mean(),
                    L1=.5*np.square(p[mask]-y[mask]).mean(),
                    L2=.5*np.square(q[:,mask]-(1-factual[None,mask])).mean(),
                    L3=.5*np.square(cf[:,mask].astype(float)-x[None,mask]).mean(),
                    proximity=np.abs(cf[:,mask]-x[None,mask]).sum(-1).mean(),
                    viol_any=np.logical_or.reduce([metrics['viol_'+n][:,mask] for n in ('race','gender','marital_status')]).mean(),
                    out_disallowed_any=(metrics['viol_workclass_unknown'][:,mask]|metrics['viol_occupation_unknown'][:,mask]).mean())
                expected['prox_norm']=expected['proximity']/29
                assert row['L4'] is None
                for key,value in expected.items(): np.testing.assert_allclose(row[key],value,rtol=2e-6,atol=1e-8)
            checked += cf.shape[0]*cf.shape[1]
            details.append(dict(setting=setting,alpha=alpha,variant=variant,seed=seed,artifact_directory=run['artifact_directory'],
                generator_file_sha256=experiment.fingerprint(path/'generator.pt'),
                candidates_file_sha256=experiment.fingerprint(path/'test_candidates.npz'),
                raw_replay_exact=True,unprojected_decoding_exact=True))
    assert len(details)==36 and checked==2930760
    assert len({(r["setting"],r["alpha"],r["seed"]) for r in details})==36
    for key, expected in summary['summary'].items():
        setting,alpha,cohort=key.split('/')
        selected=[r for r in summary['rows'] if (r['setting'],r['alpha'],r['cohort'])==(setting,float(alpha),cohort)]
        assert {r['seed'] for r in selected}==set(sweep.SEEDS)
        for metric,value in expected.items():
            if value is None:
                assert all(r[metric] is None for r in selected)
            else:
                samples=[r[metric] for r in selected]
                np.testing.assert_allclose(value['mean'],np.mean(samples),rtol=0,atol=1e-12)
                np.testing.assert_allclose(value['sd'],np.std(samples,ddof=1),rtol=0,atol=1e-12)
    result = dict(passed=True,runs=len(details),new_trained_runs=24,reused_baselines=6,reused_alpha1=6,
        decoded_candidates_audited=checked,raw_draws_replayed=checked,
        classifier_sha256=base.state_hash(classifier),classifier_accuracy=accuracy,
        checks=['Exact original data/split and classifier bytes','Matched baseline/new configuration and Generator initializations',
                'All raw samples replay exactly from saved Generator and evaluation noise',
                'Every decoded sample equals ordinary argmax/clipping; no policy projection',
                'Classifier predictions, proximity, all six policy violation rates, compliance, and full C2 rules independently recomputed',
                'Both directions and all test candidates retained',
                'Previously audited alpha=0/1 checkpoint and candidate hashes unchanged',
                'All native configurations unchanged; training histories use exactly alpha * lambda_F for added terms',
                'CounterNet-style L1/L2/L3 diagnostics, validity, accuracy and violation unions independently recomputed'],
        implementation_sha256={n:experiment.fingerprint(base.ROOT/n) for n in (
            'cf_feasibility_penalty_sweep.py','cf_feasibility_sweep_audit.py','test_cf_feasibility_sweep.py')},
        details=details)
    base.save_json(root/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=sweep.OUT)
    audit(parser.parse_args().out)
