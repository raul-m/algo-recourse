"""C2 source objective/gradient parity and conditional constraint checks."""
import ast
from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import cf_feasibility_experiment as base
from cf_feasibility_c2_experiment import OUT, C2Config, SOURCE_RANK, PAPER_RANK, c2_objective, c2_metrics


class C2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.a, cls.meta = base.load_data(OUT)
        cls.policy = base.AdultPolicy(cls.meta)
        cls.classifier = base.frozen_classifier(OUT, C2Config())
        train = torch.from_numpy(cls.a['x'][cls.a['train_indices']])
        with torch.no_grad(): p = cls.classifier(train).argmax(-1)
        cls.x = torch.cat([train[p == 0][:8], train[p == 1][:8]])

    def test_exact_upstream_c2_training_loss_and_gradients(self):
        source = OUT / 'upstream/unary-ed-const-generative-cf.py'
        tree = ast.parse(source.read_text())
        source_ranks = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'education_score' for t in n.targets))
        self.assertEqual(source_ranks, SOURCE_RANK)
        f = self.policy.by_name['education']
        namespace = dict(torch=torch, F=F, cuda=torch.device('cpu'), pred_model=self.classifier,
                         ed_dict={f['start']+i:SOURCE_RANK[c] for i,c in enumerate(f['categories'])})
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['compute_loss','train_constraint_loss']]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
        class FixedGenerator(nn.Module):
            encoded_start_cat = 2
            encoded_categorical_feature_indexes = [list(range(s,e)) for s,e in self.policy.blocks]
            def __init__(self):
                super().__init__();self.offset=nn.Parameter(torch.linspace(-2,2,29))
            def forward(self,x,c):
                raw=[torch.sigmoid(self.offset + x*.2 + k*.03) for k in range(3)]
                return dict(em=x.new_zeros(len(x),10),ev=x.new_ones(len(x),10),z=[],x_pred=raw,mc_samples=3)
        cfg=C2Config();ref=FixedGenerator();actual=FixedGenerator()
        weights={f['start']:[f['min'],f['max']] for f in self.policy.numeric}
        opt=torch.optim.SGD(ref.parameters(),lr=0)
        with redirect_stdout(io.StringIO()):
            expected=namespace['train_constraint_loss'](ref,self.x.numpy(),opt,weights,cfg.validity_weight,cfg.age_weight,cfg.margin,batch_size=len(self.x))
        out=actual(self.x,None);target=1-self.classifier(self.x).argmax(-1)
        value,_=c2_objective(torch.stack(out['x_pred']),out['em'],out['ev'],self.x,target,self.classifier,self.policy,cfg,False)
        value.backward()
        self.assertAlmostEqual(float(value.detach()),expected,places=4)
        torch.testing.assert_close(actual.offset.grad,ref.offset.grad,rtol=1e-5,atol=1e-5)

    def test_extension_retains_raw_education_and_age_terms(self):
        raw=torch.full((3,len(self.x),29),.1,requires_grad=True)
        mean=torch.zeros(len(self.x),10);var=torch.ones_like(mean)
        target=1-self.classifier(self.x).argmax(-1)
        terms=[]
        for extended in (False,True):
            loss,t=c2_objective(raw,mean,var,self.x,target,self.classifier,self.policy,C2Config(),extended)
            self.assertTrue(torch.isfinite(loss));loss.backward(retain_graph=True);terms.append(t)
        for name in ['age_penalty','education_term','constraint_term']:
            torch.testing.assert_close(terms[0][name],terms[1][name])
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(all(p.grad is None for p in self.classifier.parameters()))

    def test_full_c2_rule_including_strict_age_and_tied_ranks(self):
        x=self.x[:1].repeat(7,1).numpy();x[:,0]=.5
        f=self.policy.by_name['education'];s,e=f['start'],f['end'];cats=f['categories']
        x[:,s:e]=0;x[:,s+cats.index('Bachelors')]=1
        cf=x[None].copy()
        for i,(cat,age) in enumerate([('HS-grad',.7),('Bachelors',.5),('Masters',.5),('Masters',.6),('Assoc',.4),('Assoc',.5),('Prof-school',.6)]):
            cf[0,i,s:e]=0;cf[0,i,s+cats.index(cat)]=1;cf[0,i,0]=age
        m=c2_metrics(cf,x,np.zeros(7),np.ones((1,7)),self.policy)
        np.testing.assert_array_equal(m['c2_compliant'],[[False,True,False,True,False,True,True]])
        # Prof-school and Masters tie in source ranks, but not in paper ranks.
        x[0,s:e]=0;x[0,s+cats.index('Prof-school')]=1
        cf[0,0,s:e]=0;cf[0,0,s+cats.index('Masters')]=1
        m=c2_metrics(cf,x,np.zeros(7),np.ones((1,7)),self.policy)
        self.assertTrue(m['c2_compliant'][0,0]);self.assertFalse(m['c2_compliant_paper_rank'][0,0])

    def test_shared_classifier_data_and_split_are_identical(self):
        old,_=base.load_data(base.DEFAULT_OUT)
        for key in ['x','y','train_indices','test_indices']:np.testing.assert_array_equal(self.a[key],old[key])
        self.assertEqual(base.state_hash(self.classifier),base.state_hash(base.frozen_classifier(base.DEFAULT_OUT,base.Config())))


if __name__=='__main__':unittest.main(verbosity=2)
