"""Check that the sweep changes only the new constraints' strength."""
import unittest
from unittest.mock import patch
import torch
import cf_feasibility_experiment as base
import cf_feasibility_penalty_experiment as penalty
import cf_feasibility_penalty_sweep as sweep


class SweepTests(unittest.TestCase):
    def test_native_objective_preservation_and_alpha1_exact_parity(self):
        torch.set_num_threads(1)
        a,meta=base.load_data(base.DEFAULT_OUT)
        x=torch.from_numpy(a['x'][a['train_indices'][:8]])
        policy=base.AdultPolicy(meta)
        classifier=base.frozen_classifier(base.DEFAULT_OUT,base.Config())
        raw=torch.full((2,len(x),29),.35,requires_grad=True)
        mean=torch.zeros(len(x),10);variance=torch.ones_like(mean)
        target=1-classifier(x).argmax(-1)
        for setting in ('c1','c2'):
            cfg,_,native,_=penalty.settings(setting)
            args=(raw,mean,variance,x,target,classifier,policy,cfg,False)
            with patch.object(base.AdultPolicy,'project',penalty.forbidden_projection):
                old,old_terms=native(*args)
                alpha1,_=penalty.penalty_objective(*args)
                extra,_=penalty.added_penalties(raw,x,policy)
                for alpha in (0.,*sweep.ALPHAS):
                    loss,terms=sweep.objective(alpha)(*args)
                    expected=old+(alpha*cfg.age_weight)*extra
                    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
                    torch.testing.assert_close(torch.autograd.grad(loss,raw,retain_graph=True)[0],
                        torch.autograd.grad(expected,raw,retain_graph=True)[0],rtol=0,atol=0)
                    for key in old_terms:
                        if key!='total': torch.testing.assert_close(terms[key],old_terms[key],rtol=0,atol=0)
                    if alpha==0: torch.testing.assert_close(loss,old,rtol=0,atol=0)
                    if alpha==1:
                        torch.testing.assert_close(loss,alpha1,rtol=0,atol=0)
                        torch.testing.assert_close(torch.autograd.grad(loss,raw,retain_graph=True)[0],
                            torch.autograd.grad(alpha1,raw,retain_graph=True)[0],rtol=0,atol=0)
                self.assertTrue(all(p.grad is None for p in classifier.parameters()))


if __name__=='__main__': unittest.main()
