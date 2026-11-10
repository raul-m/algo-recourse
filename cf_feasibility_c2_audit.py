"""Audit C2 candidates independently, including rank and strict-age semantics."""
import json
from pathlib import Path

import numpy as np

from cf_feasibility_audit import audit as audit_common
from cf_feasibility_c2_experiment import OUT
from cf_feasibility_experiment import DEFAULT_OUT, save_json


def audit(out=OUT):
    audit_common(out)
    a=np.load(out/'adult_inputs.npz');old=np.load(DEFAULT_OUT/'adult_inputs.npz')
    for key in ['x','y','train_indices','test_indices']:np.testing.assert_array_equal(a[key],old[key])
    x=a['x'][a['test_indices']]
    meta=json.loads((out/'data_manifest.json').read_text())
    f=next(f for f in meta['features'] if f['name']=='education')
    factual_names=np.array(f['categories'])[x[:,f['start']:f['end']].argmax(-1)]
    # Define ranks independently from the production metric implementation.
    source_levels=[['HS-grad','School'],['Bachelors','Assoc','Some-college'],['Masters','Prof-school'],['Doctorate']]
    paper_levels=[['HS-grad','School'],['Bachelors','Assoc','Some-college'],['Masters'],['Prof-school','Doctorate']]
    summary=json.loads((out/'summary.json').read_text());counts=0
    assert len(summary['runs'])==6
    for run in summary['runs']:
        saved=np.load(out/f'{run["variant"]}_seed{run["seed"]}'/'test_candidates.npz')
        for mode,cohorts in run['cohorts'].items():
            cf=saved['cf_'+mode];pred=saved['prediction_'+mode];factual=saved['factual_prediction']
            cf_names=np.array(f['categories'])[cf[...,f['start']:f['end']].argmax(-1)]
            fixed_viol=np.zeros(cf.shape[:2],bool)
            for field in meta['features']:
                name,s,e=field['name'],field['start'],field['end']
                if name in ['race','gender','marital_status']:fixed_viol |= (cf[...,s:e]!=x[None,:,s:e]).any(-1)
                if name in ['workclass','occupation']:fixed_viol |= cf[...,s+field['categories'].index('Other/Unknown')]==1
            policy_ok=~(fixed_viol | (cf[...,0] < x[None,:,0]-1e-6))
            valid=pred!=factual[None]
            for suffix,levels in [('',source_levels),('_paper_rank',paper_levels)]:
                rank={name:i for i,names in enumerate(levels) for name in names}
                fr=np.array([rank[n] for n in factual_names])
                cr=np.array([[rank[n] for n in row] for row in cf_names])
                lower=cr<fr[None];higher=cr>fr[None]
                missing_age=higher & (cf[...,0]<=x[None,:,0])
                equal_younger=(cr==fr[None]) & (cf[...,0]<x[None,:,0])
                ok=~(lower | missing_age | equal_younger)
                metrics={'c2_compliant'+suffix:ok,'valid_c2'+suffix:valid&ok,
                    'feasible_c2'+suffix:policy_ok&ok,'valid_feasible_c2'+suffix:valid&policy_ok&ok}
                if not suffix:metrics.update(c2_education_decrease=lower,c2_higher_education_without_older_age=missing_age,c2_same_education_with_younger_age=equal_younger)
                for cohort,expected in cohorts.items():
                    mask=np.ones(len(x),bool) if cohort=='all' else factual==(0 if cohort=='low_to_high' else 1)
                    for key,values in metrics.items():np.testing.assert_allclose(values[:,mask].mean(),expected[key],rtol=1e-7,atol=1e-8)
            counts+=cf.shape[0]*cf.shape[1]
    result=json.loads((out/'audit.json').read_text())
    old_audit=json.loads((DEFAULT_OUT/'audit.json').read_text())
    assert result['classifier_sha256']==old_audit['classifier_sha256']
    assert result['paired_initializations']==old_audit['paired_initializations']
    result.update(c2_metrics_passed=True,c2_candidates_audited=counts,
        c1_split_and_classifier_identical=True,c1_initializations_identical=True,
        c2_evaluation='Repository ranks and paper-rank sensitivity; all decoded candidates; strict > age when education rank increases.',
        interpretation='The decision_attainability witnesses above satisfy only the shared six-rule policy, not necessarily C2. C2 compliance was independently recomputed for all generated candidates.')
    save_json(out/'audit.json',result)
    print('C2 audit passed:',counts,'decoded candidates; both education rankings; same C1 data/classifier/initializations.')


if __name__=='__main__':audit()
