"""Constraint signs, source-loss preservation, and no-projection smoke checks."""
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import io
import tempfile
import unittest

import numpy as np
import torch

import cf_feasibility_experiment as base
import cf_feasibility_c2_experiment as c2
import cf_feasibility_penalty_experiment as penalty


class PenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.a, cls.meta = base.load_data(base.DEFAULT_OUT)
        cls.policy = base.AdultPolicy(cls.meta)
        cls.x = torch.from_numpy(cls.a['x'][cls.a['train_indices'][:8]])
        cls.classifier = base.frozen_classifier(base.DEFAULT_OUT, base.Config())

    def test_two_sided_equality_and_one_sided_category_penalties(self):
        raw = torch.full((2, len(self.x), 29), .25, requires_grad=True)
        loss, terms = penalty.added_penalties(raw, self.x, self.policy)
        columns = [j for n in ('race','gender','marital_status')
                   for j in range(self.policy.by_name[n]['start'],self.policy.by_name[n]['end'])]
        banned = list(self.policy.banned.values())
        expected = np.abs(raw.detach().numpy()[...,columns]-self.x.numpy()[None,:,columns]).sum(-1).mean() + .5
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=5)
        loss.backward()
        expected_grad = torch.zeros_like(raw)
        expected_grad[...,columns] = torch.sign(raw.detach()[...,columns]-self.x[:,columns]) / (2*len(self.x))
        expected_grad[...,banned] = 1/(2*len(self.x))
        torch.testing.assert_close(raw.grad, expected_grad)
        self.assertEqual(len(terms), 5)

    def test_zero_extra_loss_on_allowed_candidate_and_no_duplicate_age(self):
        raw = self.x[None].clone()
        for index in self.policy.banned.values():
            raw[...,index] = 0
        raw[...,self.policy.age_index] = 0  # age remains the native term's job
        value, _ = penalty.added_penalties(raw, self.x, self.policy)
        self.assertEqual(value.item(), 0)

    def test_preserve_entire_native_c1_and_c2_loss_and_gradients(self):
        for cfg, native in [(base.Config(),base.objective),(c2.C2Config(),c2.c2_objective)]:
            raw = torch.full((2,len(self.x),29), .35, requires_grad=True)
            mean = torch.zeros(len(self.x),10)
            variance = torch.ones_like(mean)
            target = 1-self.classifier(self.x).argmax(-1)
            with patch.object(base.AdultPolicy,'project',side_effect=penalty.forbidden_projection) as blocked:
                old, before = native(raw,mean,variance,self.x,target,self.classifier,self.policy,cfg,False)
                new, after = penalty.penalty_objective(raw,mean,variance,self.x,target,self.classifier,self.policy,cfg,False)
                extra, _ = penalty.added_penalties(raw,self.x,self.policy)
                torch.testing.assert_close(new, old+cfg.age_weight*extra)
                torch.testing.assert_close(torch.autograd.grad(new,raw,retain_graph=True)[0],
                    torch.autograd.grad(old+cfg.age_weight*extra,raw)[0])
                for key in before:
                    if key != 'total': torch.testing.assert_close(before[key],after[key])
                blocked.assert_not_called()
            self.assertTrue(all(p.grad is None for p in self.classifier.parameters()))

    def test_actual_training_and_evaluation_never_project(self):
        a = dict(self.a)
        a['train_indices'] = a['train_indices'][:8]
        a['test_indices'] = a['test_indices'][:5]
        for setting in ('c1','c2'):
            cfg, source, _, metric = penalty.settings(setting)
            cfg = replace(cfg,epochs=1,batch=8,mc_samples=2,eval_samples=2,eval_batch=5)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                c2.copy_identical(source/'classifier.pt',out/'classifier.pt')
                c2.copy_identical(source/'data_manifest.json',out/'data_manifest.json')
                with patch.object(base,'load_data',return_value=(a,self.meta)), \
                     patch.object(base.AdultPolicy,'project',side_effect=penalty.forbidden_projection) as blocked, \
                     redirect_stdout(io.StringIO()):
                    base.train_generator(out,cfg,penalty.VARIANT,42,objective_fn=penalty.penalty_objective)
                    result = base.evaluate_generator(out,cfg,penalty.VARIANT,42,metrics_fn=metric)
                    blocked.assert_not_called()
                self.assertEqual(list(result['cohorts']),['decoded'])
                candidates = np.load(out/'loss_only_seed42/test_candidates.npz')
                self.assertEqual(candidates['cf_decoded'].shape,(2,5,29))
                np.testing.assert_array_equal(base.AdultPolicy(self.meta).decode(torch.from_numpy(candidates['cf_native_soft'])).numpy(),candidates['cf_decoded'])


if __name__ == '__main__':
    unittest.main()
