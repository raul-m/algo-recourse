"""Known-value checks for the CounterNet-style reporting diagnostics."""
import math
import unittest

import numpy as np
import torch

import cf_feasibility_experiment as base
from cf_feasibility_comparison_metrics import summarize
from cf_feasibility_comparison_report import format_cell


class FixedClassifier(torch.nn.Module):
    def forward(self,x):
        result=x.new_zeros((*x.shape[:-1],2))
        result[...,0]=math.log(3)  # P(class 1)=1/4, predict class 0
        return result


class ComparisonTests(unittest.TestCase):
    def test_overlapping_violations_and_squared_error_diagnostics(self):
        _,meta=base.load_data(base.DEFAULT_OUT)
        by_name={f['name']:f for f in meta['features']}
        x=np.zeros((3,29),dtype=np.float32);x[:,:2]=.6
        for f in meta['features']:
            if f['categorical']:x[:,f['start']]=1
        cf=np.broadcast_to(x,(2,3,29)).copy()
        for name,row,category in [('race',0,1),('gender',0,1),('marital_status',1,1),('workclass',0,'Other/Unknown'),('occupation',0,'Other/Unknown')]:
            f=by_name[name];j=f['categories'].index(category) if isinstance(category,str) else category
            cf[:,row,f['start']:f['end']]=0;cf[:,row,f['start']+j]=1
        cf[:,2,0]=.1
        m=summarize(x,np.array([0,1,1]),cf,FixedClassifier(),meta['features'],np.ones(3,bool))
        self.assertAlmostEqual(m['viol_any'],2/3)
        self.assertAlmostEqual(m['out_disallowed_any'],1/3)
        self.assertAlmostEqual(m['viol_age_decrease'],1/3)
        self.assertAlmostEqual(m['acc'],1/3)
        self.assertEqual(m['validity'],0)
        self.assertAlmostEqual(m['L1'],.5*(.25**2+2*.75**2)/3)
        self.assertAlmostEqual(m['L2'],.5*.75**2)
        self.assertAlmostEqual(m['L3'],.5*10.25/(3*29))
        self.assertAlmostEqual(m['proximity'],3.5)
        self.assertAlmostEqual(m['prox_norm'],3.5/29)
        self.assertIsNone(m['L4'])

    def test_rare_violations_are_not_displayed_as_zero(self):
        self.assertIn('&lt;0.1%',format_cell(dict(mean=1/244230,sd=0),'viol_gender'))
        self.assertNotIn('&lt;',format_cell(dict(mean=0,sd=0),'viol_gender'))
        self.assertIn('>—<',format_cell(None,'L4'))


if __name__=='__main__':
    unittest.main()
