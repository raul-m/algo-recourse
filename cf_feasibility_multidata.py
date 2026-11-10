"""C1/C2-inspired Model-approx adaptations and constraint sweeps on Credit/OULAD.

These are new dataset adaptations, not published Credit/OULAD C1/C2 baselines.
Preserve the Adult C1/C2 settings and native reconstruction convention explicitly.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

import cf_feasibility_experiment as base
from cf_feasibility_c2_experiment import C2Config

ROOT=base.ROOT
OUT=ROOT/'trained_models/cf_feasibility_multidata'
ALPHAS=(0.,.1,.3,1.,3.,10.)
SEEDS=(42,43,44)


def configuration(setting):
    return base.Config() if setting=='c1' else C2Config()


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(out):
    with np.load(out/'inputs.npz') as saved:
        a={k:saved[k] for k in saved.files}
    meta=json.loads((out/'data_manifest.json').read_text())
    for key,field in [('x','encoded_x_sha256'),('y','encoded_y_sha256'),('train_indices','train_indices_sha256'),('test_indices','test_indices_sha256')]:
        assert hashlib.sha256(a[key].tobytes()).hexdigest()==meta[field]
    return a,meta


class Policy:
    def __init__(self,meta):
        self.features=meta['features']
        self.by_name={f['name']:f for f in self.features}
        self.rules=meta['policy']
        self.fixed=self.rules['fixed']
        self.numeric=[f for f in self.features if not f['categorical']]
        self.blocks=[(f['start'],f['end']) for f in self.features if f['categorical']]
        self.category_sum_matrix=np.zeros((meta['dimensions'],len(self.blocks)),dtype=np.float32)
        for index,(s,e) in enumerate(self.blocks): self.category_sum_matrix[s:e,index]=1
        self.banned={name:[self.by_name[name]['start']+self.by_name[name]['categories'].index(c) for c in cats]
                     for name,cats in self.rules['banned'].items()}
        self.age=self.by_name[self.rules['age']]
        self.dataset=meta['dataset']
        self.education_name='EDUCATION' if self.dataset=='credit' else 'highest_education' if self.dataset=='oulad' else 'education'
        self.education_ranks=({'3':0,'2':1,'1':2} if self.dataset=='credit' else
            {'No Formal quals':0,'Lower Than A Level':1,'A Level or Equivalent':2,'HE Qualification':3,'Post Graduate Qualification':4} if self.dataset=='oulad' else {})
        assert [f['start'] for f in self.numeric]==list(range(len(self.numeric)))
        if self.rules['age_kind']=='ordinal':
            assert self.age['categories']==self.rules['age_order']

    def project(self,*args,**kwargs):
        raise AssertionError('Policy projection is forbidden in these experiments')

    def decode(self,raw):
        parts=[]
        for f in self.features:
            block=raw[...,f['start']:f['end']]
            parts.append(F.one_hot(block.argmax(-1),block.shape[-1]).to(raw.dtype) if f['categorical'] else block.clamp(0,1))
        return torch.cat(parts,-1)

    def age_penalty(self,raw,x):
        s,e=self.age['start'],self.age['end']
        if self.rules['age_kind']=='continuous':
            return F.relu(x[...,s]-raw[...,s]).mean()
        # Each score for a band younger than the factual band is discouraged.
        # No band is masked, renormalized, or copied into the generated vector.
        lower=torch.arange(e-s,device=raw.device)<x[...,s:e].argmax(-1)[...,None]
        return (F.relu(raw[...,s:e])*lower).sum(-1).mean()

    def added_penalties(self,raw,x):
        terms={}
        fixed_indices=[j for name in self.fixed for j in range(self.by_name[name]['start'],self.by_name[name]['end'])]
        fixed_delta=(raw[...,fixed_indices]-x[...,fixed_indices]).abs()
        offset=0
        for name in self.fixed:
            f=self.by_name[name]
            width=f['end']-f['start']
            terms['penalty_'+name]=fixed_delta[...,offset:offset+width].sum(-1).mean()
            offset+=width
        banned_indices=[j for indices in self.banned.values() for j in indices]
        banned_scores=F.relu(raw[...,banned_indices])
        offset=0
        for name,indices in self.banned.items():
            terms['penalty_banned_'+name]=banned_scores[...,offset:offset+len(indices)].sum(-1).mean()
            offset+=len(indices)
        return fixed_delta.sum(-1).mean()+banned_scores.sum(-1).mean(),terms

    def education_change(self,raw,x):
        f=self.by_name[self.education_name];s,e=f['start'],f['end']
        if self.dataset=='credit':
            # Only documented, ordered levels 3 < 2 < 1 can be compared.
            # Codes 0/4/5/6 are not assigned a fictitious education rank.
            known=raw.new_tensor([c in self.education_ranks for c in f['categories']])
            ranks=raw.new_tensor([self.education_ranks.get(c,0) for c in f['categories']])
            factual_known=(x[...,s:e]*known).sum(-1)
            factual_rank=(x[...,s:e]*ranks).sum(-1)
            changes=(raw[...,s:e]*known*(ranks-factual_rank[...,None])).sum(-1)
            return (changes*factual_known).mean()
        ranks=raw.new_tensor([self.education_ranks[c] for c in f['categories']])
        return ((raw[...,s:e]*ranks).sum(-1)-(x[...,s:e]*ranks).sum(-1)).mean()


def objective(raw,mean,variance,x,target,classifier,policy,cfg,alpha):
    # Same released s:-1 convention; final categorical coordinate is omitted.
    n=len(policy.numeric)
    cat_distance=(raw[...,n:-1]-x[...,n:-1]).abs().sum(-1)
    ranges=raw.new_tensor([f['max']-f['min'] for f in policy.numeric])
    proximity=cat_distance+((raw[...,:n]-x[...,:n]).abs()*ranges).sum(-1)
    simplex=(1-raw@raw.new_tensor(policy.category_sum_matrix)).abs().sum(-1)
    kl=.5*(mean.square()+variance-variance.log()-1).mean()
    logits=classifier(raw)
    hinge=raw.new_zeros(())
    for label in (0,1):
        selected=target==label
        if selected.any():
            difference=logits[:,selected,label].sigmoid()-logits[:,selected,1-label].sigmoid()
            hinge=hinge+F.relu(cfg.margin-difference).mean()
    age=policy.age_penalty(raw,x)
    extra,details=policy.added_penalties(raw,x)
    native=(proximity+simplex).mean()+kl+cfg.validity_weight*hinge+cfg.age_weight*age
    education=raw.new_zeros(())
    if isinstance(cfg,C2Config):
        education=-cfg.education_coefficient*policy.education_change(raw,x)
        native=native+cfg.age_weight*education
    weighted=alpha*cfg.age_weight*extra
    total=native+weighted
    return total,dict(total=total,native_total=native,reconstruction=proximity.mean(),simplex=simplex.mean(),
        kl=kl,validity_hinge=hinge,age_penalty=age,education_term=education,added_penalty=extra,added_weighted_penalty=weighted,**details)


def prepare(out,cfg):
    _,meta=load_data(out)
    policy=Policy(meta)
    protocol=dict(dataset=meta['dataset'],configs={s:asdict(configuration(s)) for s in ('c1','c2')},alphas=list(ALPHAS),seeds=list(SEEDS),
        dimensions=meta['dimensions'],data_sha256=meta['encoded_x_sha256'],projection=False,
        baseline='C1-inspired age-only and C2-inspired age/education adaptations; not published baselines on these datasets.',
        losses=dict(c1='L_base + 29 H_0.764 + 192 A_dataset + alpha * 192 P_extra',
            c2='L_base + 76 H_0.344 + 87 A_dataset - (87 * .05) D_dataset + alpha * 87 P_extra'),
        age_loss='mean relu(x_age-r_age)' if meta['policy']['age_kind']=='continuous' else 'mean sum_j[rank(j)<rank(x_age)] relu(r_age,j)',
        extra_loss='Mean across people and MC draws of summed fixed-block absolute coordinate differences plus forbidden-category sigmoid scores.',
        reconstruction='Released categorical slice numeric_count:-1 and raw-range-weighted numerical absolute differences. No loss rescaling.',
        computation='Vectorized sums and categorical block-sum matrix; same mathematical loss and gradients, with floating-point summation order differences. All final Generator runs use this implementation.',
        coefficient_scope='Adult C1/C2 coefficients transferred without dataset-specific tuning; these are not optimality claims. Large raw numerical ranges can dominate this objective.',
        education_ranks=policy.education_ranks,
        education_loss='mean I_known(x) * sum_k_in_known r_k*(rank(k)-rank(x)); unranked factual or generated categories contribute no education comparison' if meta['dataset']=='credit' else 'mean [sum_k r_k*rank(k) - rank(x)], using all five education levels',
        c2_scope='Credit education codes 0/4/5/6 are unranked, so the education incentive is partial. OULAD has ordinal age bands, so a strict actual-age increase within a band cannot be observed. Neither soft objective enforces the complete original C2 verbal rule.',
        categorical_omitted_coordinate=meta['encoded_names'][-1],
        classifier=f'{meta["dimensions"]} -> 10 -> 2, no activation, trained once on the exact CounterNet training indices, then frozen; separate from CounterNet Predictor.',
        generator=f'{meta["dimensions"]+1} -> 20 -> 16 -> 14 -> 12 -> 10 latent; decoder 11 -> 12 -> 14 -> 16 -> 20 -> {meta["dimensions"]}; original BN/dropout/sigmoid architecture.',
        target='Opposite predicted class in both directions, for all rows. Credit class 1 means default; OULAD class 1 means Pass/Distinction.',
        evaluation='Ten draws per test row, all retained; categorical argmax and numeric [0,1] bounds only. Final epoch, no early stopping, test selection or best-of candidates.',
        policy=meta['policy'],upstream_commit=base.UPSTREAM_COMMIT,
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),
        implementation_sha256={n:fingerprint(ROOT/n) for n in ('cf_feasibility_multidata.py','cf_feasibility_multidata_prepare.py','cf_feasibility_experiment.py','cf_feasibility_c2_experiment.py')})
    path=out/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol,'Protocol differs; use a separate experiment directory'
    else: base.save_json(path,protocol)


def train_classifier(out,cfg):
    a,meta=load_data(out)
    if (out/'classifier.pt').exists(): return
    base.seed_all(cfg.classifier_seed)
    model=base.BlackBox(meta['dimensions'])
    train=a['train_indices'];x=torch.from_numpy(a['x'][train]);y=torch.from_numpy(a['y'][train])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.classifier_lr)
    rng=np.random.default_rng(cfg.classifier_seed)
    history=[];started=time.perf_counter()
    for epoch in range(1,cfg.classifier_epochs+1):
        model.train();losses=[]
        for indices in np.array_split(rng.permutation(len(x)),len(x)//cfg.classifier_batch):
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x[indices]),y[indices])
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,train_cross_entropy=float(np.mean(losses))))
        if epoch==1 or epoch%20==0: print(f'{meta["dataset"]} classifier {epoch}/{cfg.classifier_epochs} loss={np.mean(losses):.5f}',flush=True)
    model.eval()
    with torch.no_grad(): predicted=model(torch.from_numpy(a['x'])).argmax(-1).numpy()
    results=dict(config=asdict(cfg),state_sha256=base.state_hash(model),history=history,seconds=time.perf_counter()-started)
    for split in ('train','test'):
        idx=a[split+'_indices'];truth=a['y'][idx];pred=predicted[idx]
        results[split]=dict(n=len(idx),accuracy=float((truth==pred).mean()),
            class_counts=[int((truth==k).sum()) for k in (0,1)],prediction_counts=[int((pred==k).sum()) for k in (0,1)],
            confusion=[[int(((truth==i)&(pred==j)).sum()) for j in (0,1)] for i in (0,1)])
    torch.save(dict(state_dict=model.state_dict(),config=asdict(cfg),data_sha256=meta['encoded_x_sha256'],dimensions=meta['dimensions']),out/'classifier.pt')
    base.save_json(out/'classifier_results.json',results)
    print(json.dumps(results['test']),flush=True)


def frozen_classifier(out,cfg):
    _,meta=load_data(out)
    saved=torch.load(out/'classifier.pt',map_location='cpu',weights_only=True)
    assert saved['data_sha256']==meta['encoded_x_sha256'] and saved['dimensions']==meta['dimensions']
    for key in ('classifier_seed','classifier_epochs','classifier_lr','classifier_batch'):
        assert saved['config'][key]==getattr(cfg,key)
    model=base.BlackBox(meta['dimensions']);model.load_state_dict(saved['state_dict'])
    return model.eval().requires_grad_(False)


def run_path(out,alpha,seed,setting='c1'):
    return out/setting/f'alpha_{alpha:g}'/f'seed_{seed}'


def train(out,cfg,alpha,seed,setting='c1'):
    assert isinstance(cfg,C2Config)==(setting=='c2')
    path=run_path(out,alpha,seed,setting);path.mkdir(parents=True,exist_ok=True)
    if (path/'generator.pt').exists():
        print(f'Reusing {out.name} {setting} alpha={alpha:g} seed={seed}',flush=True);return
    a,meta=load_data(out);policy=Policy(meta);classifier=frozen_classifier(out,cfg)
    clf_hash=base.state_hash(classifier)
    x=torch.from_numpy(a['x'][a['train_indices']])
    with torch.no_grad(): target=1-classifier(x).argmax(-1)
    base.seed_all(seed)
    model=base.ModelApproxCF(meta['dimensions'],cfg.latent_dim)
    initial=base.state_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    rng=np.random.default_rng(seed);history=[];started=time.perf_counter();model.train()
    for epoch in range(1,cfg.epochs+1):
        order=rng.permutation(len(x));terms=[]
        for offset in range(0,len(x),cfg.batch):
            ix=order[offset:offset+cfg.batch];batch=x[ix];labels=target[ix]
            optimizer.zero_grad(set_to_none=True)
            raw,mean,var=model(batch,labels,cfg.mc_samples)
            loss,stats=objective(raw,mean,var,batch,labels,classifier,policy,cfg,alpha)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Non-finite gradient')
            optimizer.step();terms.append({k:float(v.detach()) for k,v in stats.items()})
        row=dict(epoch=epoch,elapsed_seconds=time.perf_counter()-started,**{k:float(np.mean([t[k] for t in terms])) for k in terms[0]})
        history.append(row);base.save_json(path/'history.json',history)
        if epoch==1 or epoch%5==0: print(f'{out.name} {setting} alpha={alpha:g} seed={seed} epoch={epoch}/{cfg.epochs} loss={row["total"]:.4f} recon={row["reconstruction"]:.4f} hinge={row["validity_hinge"]:.4f} elapsed={row["elapsed_seconds"]:.1f}s',flush=True)
    assert base.state_hash(classifier)==clf_hash
    saved=dict(state_dict=model.state_dict(),config=asdict(cfg),dataset=meta['dataset'],setting=setting,dimensions=meta['dimensions'],alpha=alpha,seed=seed,
        classifier_sha256=clf_hash,initial_sha256=initial,data_sha256=meta['encoded_x_sha256'])
    torch.save(saved,path/'generator.pt')
    base.save_json(path/'training.json',dict(alpha=alpha,seed=seed,initial_sha256=initial,final_sha256=base.state_hash(model),
        classifier_sha256=clf_hash,seconds=time.perf_counter()-started,epochs=cfg.epochs))


def metrics(cf,x,y,classifier,policy,mask):
    x=x[mask];y=y[mask];cf=cf[:,mask]
    with torch.no_grad():
        logits=classifier(torch.from_numpy(x));cf_logits=classifier(torch.from_numpy(cf))
        factual=logits.argmax(-1).numpy();predicted=cf_logits.argmax(-1).numpy()
        p=logits.softmax(-1)[:,1].numpy().astype(float);q=cf_logits.softmax(-1)[...,1].numpy().astype(float)
    distance=np.abs(cf-x[None]).sum(-1)
    values=dict(n_test=len(x),draws_per_person=cf.shape[0],acc=float((factual==y).mean()),validity=float((predicted!=factual[None]).mean()),
        L1=float(.5*np.square(p-y).mean()),L2=float(.5*np.square(q-(1-factual[None])).mean()),
        L3=float(.5*np.square(cf.astype(float)-x[None]).mean()),L4=None,proximity=float(distance.mean()),prox_norm=float(distance.mean()/x.shape[-1]))
    changed=[]
    for name in policy.fixed:
        f=policy.by_name[name]
        flag=np.abs(cf[...,f['start']:f['end']]-x[None,:,f['start']:f['end']]).sum(-1)>1e-6
        values['viol_'+name]=float(flag.mean());changed.append(flag)
    values['viol_any']=float(np.logical_or.reduce(changed).mean())
    s,e=policy.age['start'],policy.age['end']
    age_flag=cf[...,s]<x[None,:,s]-1e-6 if policy.rules['age_kind']=='continuous' else cf[...,s:e].argmax(-1)<x[None,:,s:e].argmax(-1)
    values['viol_age_decrease']=float(age_flag.mean())
    forbidden=[]
    for name,indices in policy.banned.items():
        flag=(cf[...,indices]>.5).any(-1);forbidden.append(flag)
        values['out_disallowed_'+name]=float(flag.mean())
    values['out_disallowed_any']=float(np.logical_or.reduce(forbidden).mean())
    return values


def evaluate(out,cfg,alpha,seed,setting='c1'):
    path=run_path(out,alpha,seed,setting)
    if (path/'evaluation.json').exists(): return
    a,meta=load_data(out);policy=Policy(meta);classifier=frozen_classifier(out,cfg)
    saved=torch.load(path/'generator.pt',map_location='cpu',weights_only=True)
    assert saved['config']==asdict(cfg) and saved['alpha']==alpha and saved['seed']==seed and saved['setting']==setting
    assert saved['classifier_sha256']==base.state_hash(classifier)
    model=base.ModelApproxCF(meta['dimensions'],cfg.latent_dim).eval();model.load_state_dict(saved['state_dict'])
    x=a['x'][a['test_indices']];y=a['y'][a['test_indices']]
    rng=torch.Generator().manual_seed(cfg.eval_seed+seed);raws=[]
    with torch.no_grad():
        factual=classifier(torch.from_numpy(x)).argmax(-1).numpy()
        for offset in range(0,len(x),cfg.eval_batch):
            batch=torch.from_numpy(x[offset:offset+cfg.eval_batch]);target=1-classifier(batch).argmax(-1)
            mean,var=model.encode(batch,target)
            z=mean[None]+var.sqrt()[None]*torch.randn(cfg.eval_samples,len(batch),cfg.latent_dim,generator=rng)
            label=target.float()[None,:,None].expand(cfg.eval_samples,-1,-1)
            raw=model.decoder_mean(torch.cat([z,label],-1).reshape(-1,cfg.latent_dim+1)).reshape(cfg.eval_samples,len(batch),meta['dimensions'])
            raws.append(raw.numpy())
        raw=np.concatenate(raws,axis=1)
        cf=policy.decode(torch.from_numpy(raw)).numpy()
        raw_pred=classifier(torch.from_numpy(raw)).argmax(-1).numpy()
        pred=classifier(torch.from_numpy(cf)).argmax(-1).numpy()
    cohorts={}
    for cohort,mask in [('all',np.ones(len(x),bool)),('0_to_1',factual==0),('1_to_0',factual==1)]:
        if not mask.any():
            cohorts[cohort]=dict(n_test=0);continue
        values=metrics(cf,x,y,classifier,policy,mask)
        values['raw_validity']=float((raw_pred[:,mask]!=factual[None,mask]).mean())
        cohorts[cohort]=values
    np.savez_compressed(path/'test_candidates.npz',test_indices=a['test_indices'],factual_prediction=factual,
        cf_native_soft=raw,cf_decoded=cf,prediction_decoded=pred,prediction_native_soft=raw_pred)
    base.save_json(path/'evaluation.json',dict(dataset=meta['dataset'],setting=setting,alpha=alpha,added_weight=alpha*cfg.age_weight,seed=seed,cohorts=cohorts))
    print('EVAL '+out.name+' '+setting+' '+json.dumps(dict(alpha=alpha,seed=seed,**cohorts['all'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['classifier','train','all'])
    parser.add_argument('--dataset',choices=['credit','oulad'],required=True)
    parser.add_argument('--setting',choices=['c1','c2'])
    parser.add_argument('--alpha',type=float,choices=ALPHAS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=OUT/args.dataset;cfg=base.Config();prepare(out,cfg)
    if args.command in ('classifier','all'): train_classifier(out,cfg)
    if args.command=='classifier': return
    for setting in ([args.setting] if args.setting else ['c1','c2']):
        cfg=configuration(setting)
        for alpha in ([args.alpha] if args.alpha is not None else ALPHAS):
            for seed in ([args.seed] if args.seed is not None else SEEDS):
                train(out,cfg,alpha,seed,setting);evaluate(out,cfg,alpha,seed,setting)


if __name__=='__main__': main()
