"""Representation, native-loss parity, ordinal constraints and full-loop checks."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_penalty_experiment as adult_penalty
import cf_feasibility_multidata as experiment
from cf_feasibility_multidata_results import independent_metrics


class MultiDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_exported_splits_and_feature_dimensions(self):
        for dataset,shape,n_train in [('credit',(30000,33),22500),('oulad',(32593,127),24444)]:
            a,meta=experiment.load_data(experiment.OUT/dataset)
            self.assertEqual(a['x'].shape,shape);self.assertEqual(len(a['train_indices']),n_train)
            self.assertEqual(len(np.intersect1d(a['train_indices'],a['test_indices'])),0)
            np.testing.assert_array_equal(np.sort(np.r_[a['train_indices'],a['test_indices']]),np.arange(shape[0]))
            policy=experiment.Policy(meta)
            np.testing.assert_array_equal(policy.decode(torch.from_numpy(a['x'])).numpy(),a['x'])

    def test_adult_loss_and_gradient_parity_at_all_alphas(self):
        a,meta=base.load_data(base.DEFAULT_OUT)
        meta={**meta,'dataset':'adult','policy':dict(fixed=['race','gender','marital_status'],age='age',age_kind='continuous',
            age_order=None,banned={'workclass':['Other/Unknown'],'occupation':['Other/Unknown']})}
        policy=experiment.Policy(meta);original_policy=base.AdultPolicy(meta)
        x=torch.from_numpy(a['x'][a['train_indices'][:8]])
        raw=torch.full((2,len(x),29),.35,requires_grad=True)
        mean=torch.zeros(len(x),10);var=torch.ones_like(mean)
        cfg=base.Config();clf=base.frozen_classifier(base.DEFAULT_OUT,cfg);target=1-clf(x).argmax(-1)
        native,_=base.objective(raw,mean,var,x,target,clf,original_policy,cfg,False)
        added,_=adult_penalty.added_penalties(raw,x,original_policy)
        for alpha in experiment.ALPHAS:
            actual,_=experiment.objective(raw,mean,var,x,target,clf,policy,cfg,alpha)
            expected=native+alpha*cfg.age_weight*added
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-5)
            torch.testing.assert_close(torch.autograd.grad(actual,raw,retain_graph=True)[0],torch.autograd.grad(expected,raw,retain_graph=True)[0],rtol=1e-6,atol=1e-6)

    def test_ordinal_age_loss_and_decoding_do_not_project(self):
        a,meta=experiment.load_data(experiment.OUT/'oulad');policy=experiment.Policy(meta)
        x=torch.from_numpy(a['x'][:3]).clone();s,e=policy.age['start'],policy.age['end']
        x[:,s:e]=torch.eye(3)
        raw=x[None].clone();raw[...,s:e]=torch.tensor([.6,.3,.1]);raw.requires_grad_()
        value=policy.age_penalty(raw,x)
        self.assertAlmostEqual(value.item(),(0+.6+.9)/3,places=6)
        gradient=torch.autograd.grad(value,raw)[0]
        torch.testing.assert_close(gradient[0,:,s:e],torch.tensor([[0,0,0],[1/3,0,0],[1/3,1/3,0]]))
        decoded=policy.decode(raw)
        self.assertEqual(decoded[0,2,s:e].argmax().item(),0)  # decreasing age survives decoding
        with self.assertRaises(AssertionError): policy.project(raw,x)

    def test_extra_constraints_have_no_age_term_and_use_all_banned_codes(self):
        for dataset in ('credit','oulad'):
            a,meta=experiment.load_data(experiment.OUT/dataset);policy=experiment.Policy(meta)
            x=torch.from_numpy(a['x'][:4]);raw=torch.full((2,4,meta['dimensions']),.25,requires_grad=True)
            loss,terms=policy.added_penalties(raw,x)
            expected=0.
            for name in policy.fixed:
                f=policy.by_name[name]
                expected+=(raw[...,f['start']:f['end']]-x[...,f['start']:f['end']]).abs().sum(-1).mean()
            expected+=.25*sum(len(ix) for ix in policy.banned.values())
            torch.testing.assert_close(loss,expected)
            grad=torch.autograd.grad(loss,raw)[0]
            s,e=policy.age['start'],policy.age['end']
            self.assertEqual(grad[...,s:e].abs().sum().item(),0)
            self.assertEqual(len(terms),len(policy.fixed)+len(policy.banned))

    def test_c2_education_conversion_and_unranked_credit_codes(self):
        a,meta=experiment.load_data(experiment.OUT/'credit');policy=experiment.Policy(meta)
        f=policy.by_name['EDUCATION'];s,e=f['start'],f['end']
        x=torch.from_numpy(a['x'][:1]).clone();x[:,s:e]=0;x[:,s+f['categories'].index('3')]=1
        for category,expected in [('1',2),('2',1),('3',0),('0',0),('4',0),('5',0),('6',0)]:
            raw=x[None].clone();raw[...,s:e]=0;raw[...,s+f['categories'].index(category)]=1
            self.assertEqual(policy.education_change(raw,x).item(),expected)
        x[:,s:e]=0;x[:,s+f['categories'].index('4')]=1
        self.assertEqual(policy.education_change(torch.ones_like(x)[None],x).item(),0)
        a,meta=experiment.load_data(experiment.OUT/'oulad');policy=experiment.Policy(meta)
        f=policy.by_name['highest_education'];s,e=f['start'],f['end']
        x=torch.from_numpy(a['x'][:1]).clone();x[:,s:e]=0;x[:,s+f['categories'].index('No Formal quals')]=1
        raw=x[None].clone();raw[...,s:e]=0;raw[...,s+f['categories'].index('Post Graduate Qualification')]=1
        self.assertEqual(policy.education_change(raw,x).item(),4)

    def test_train_evaluate_smoke_and_no_projection(self):
        for dataset in ('credit','oulad'):
            a,meta=experiment.load_data(experiment.OUT/dataset)
            a={**a,'train_indices':a['train_indices'][:8],'test_indices':a['test_indices'][:5]}
            cfg=replace(base.Config(),classifier_epochs=1,classifier_batch=4,epochs=1,batch=8,mc_samples=2,eval_samples=2,eval_batch=5)
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                with patch.object(experiment,'load_data',return_value=(a,meta)),patch.object(experiment.Policy,'project',side_effect=AssertionError('projection called')) as blocked,redirect_stdout(io.StringIO()):
                    experiment.train_classifier(out,cfg)
                    for setting in ('c1','c2'):
                        chosen=replace(experiment.configuration(setting),classifier_epochs=1,classifier_batch=4,epochs=1,batch=8,mc_samples=2,eval_samples=2,eval_batch=5)
                        experiment.train(out,chosen,.3,42,setting);experiment.evaluate(out,chosen,.3,42,setting)
                    blocked.assert_not_called()
                with np.load(experiment.run_path(out,.3,42)/'test_candidates.npz') as values:
                    self.assertEqual(values['cf_decoded'].shape,(2,5,meta['dimensions']))
                    np.testing.assert_array_equal(values['cf_decoded'],experiment.Policy(meta).decode(torch.from_numpy(values['cf_native_soft'])).numpy())

    def test_independent_metrics_with_overlapping_constraint_failures(self):
        for dataset in ('credit','oulad'):
            a,meta=experiment.load_data(experiment.OUT/dataset);policy=experiment.Policy(meta)
            x=a['x'][:3].copy();y=a['y'][:3]
            s,e=policy.age['start'],policy.age['end']
            if policy.rules['age_kind']=='continuous': x[:,s]=.5
            else: x[:,s:e]=np.eye(e-s,dtype=np.float32)[-1]
            cf=np.broadcast_to(x,(2,*x.shape)).copy()
            def change(draw,row,name):
                f=policy.by_name[name];start,end=f['start'],f['end']
                j=(cf[draw,row,start:end].argmax()+1)%(end-start)
                cf[draw,row,start:end]=0;cf[draw,row,start+j]=1
            change(0,0,policy.fixed[0]);change(0,1,policy.fixed[0]);change(0,1,policy.fixed[1]);change(1,1,policy.fixed[-1])
            if policy.rules['age_kind']=='continuous': cf[1,2,s]=.4
            else: cf[1,2,s:e]=np.eye(e-s,dtype=np.float32)[0]
            name=next(iter(policy.banned));f=policy.by_name[name];j=policy.banned[name][0]
            for draw in (0,1): cf[draw,2,f['start']:f['end']]=0;cf[draw,2,j]=1
            raw=cf*.8+.1
            clf=experiment.frozen_classifier(experiment.OUT/dataset,base.Config());mask=np.ones(3,bool)
            actual=experiment.metrics(cf,x,y,clf,policy,mask)
            independent=independent_metrics(cf,raw,x,y,clf,meta['features'],meta['policy'],mask)
            for key,value in actual.items():
                if value is None: self.assertIsNone(independent[key])
                else: np.testing.assert_allclose(value,independent[key],rtol=2e-6,atol=1e-8)
            self.assertAlmostEqual(actual['viol_any'],.5)
            self.assertAlmostEqual(actual['viol_age_decrease'],1/6)
            self.assertAlmostEqual(actual['out_disallowed_any'],2/6)


if __name__=='__main__': unittest.main()
