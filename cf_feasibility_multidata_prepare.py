"""Export exact saved Credit/OULAD CounterNet inputs, without refitting or splitting."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/cf-feasibility-matplotlib')
import numpy as np
from relax.data_module import DataModule

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'trained_models/cf_feasibility_multidata'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dataset):
    source=ROOT/'relax-assets'/dataset/'data'
    grid=ROOT/('cred_grid.py' if dataset=='credit' else 'oulad_grid.py')
    constants={}
    for node in ast.parse(grid.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('FIXED','AGE','BANNED'):
                constants[node.targets[0].id]=ast.literal_eval(node.value)
    dm=DataModule.load_from_path(source)
    x=np.asarray(dm.xs,dtype=np.float32)
    y=np.asarray(dm.ys).reshape(-1).astype(np.int64)
    train=np.asarray(dm.train_indices,dtype=np.int64)
    test=np.asarray(dm.test_indices,dtype=np.int64)
    cfg=json.loads((source/'config.json').read_text())
    np.testing.assert_array_equal(train,cfg['train_indices'])
    np.testing.assert_array_equal(test,cfg['test_indices'])
    assert len(np.intersect1d(train,test))==0
    np.testing.assert_array_equal(np.sort(np.r_[train,test]),np.arange(len(x)))
    assert np.isfinite(x).all() and set(np.unique(y))=={0,1}
    target='Y' if dataset=='credit' else 'final_result'
    np.testing.assert_array_equal(y,dm.data[target].to_numpy())
    features,encoded_names=[],[]
    for f,(s,e) in zip(dm.features,dm.features.feature_indices):
        tr=f.transformation.transformer
        item=dict(name=f.name,start=int(s),end=int(e),categorical=bool(f.is_categorical))
        if f.is_categorical:
            item['categories']=[str(c) for c in np.asarray(tr.categories_).reshape(-1)]
            assert np.isin(x[:,s:e],[0,1]).all()
            np.testing.assert_array_equal(x[:,s:e].sum(1),np.ones(len(x)))
            encoded_names.extend(f'{f.name}_{c}' for c in item['categories'])
        else:
            item['min']=float(np.asarray(tr.min_).reshape(-1)[0])
            item['max']=float(np.asarray(tr.max_).reshape(-1)[0])
            encoded_names.append(f.name)
        features.append(item)
    policy=dict(fixed=constants['FIXED'],age=constants['AGE'],
        age_kind='continuous' if dataset=='credit' else 'ordinal',
        age_order=None if dataset=='credit' else ['0-35','35-55','55<='],
        banned={name:[str(v) for v in values] for name,values in constants['BANNED'].items()})
    by_name={f['name']:f for f in features}
    assert all(name in by_name for name in policy['fixed']+[policy['age']])
    for name,categories in policy['banned'].items():
        assert set(categories)<=set(by_name[name]['categories'])
    if policy['age_kind']=='ordinal':
        assert by_name[policy['age']]['categories']==policy['age_order']
    files=['config.json','data.csv','features/data.npy','features/treedef.json','label/data.npy','label/treedef.json']
    metadata=dict(dataset=dataset,rows=len(x),dimensions=x.shape[1],train_rows=len(train),test_rows=len(test),
        target=target,label_semantics=['No default','Default next month'] if dataset=='credit' else ['Fail/Withdrawn','Pass/Distinction'],
        features=features,encoded_names=encoded_names,policy=policy,
        source_directory=str(source.relative_to(ROOT)),source_sha256={f:fingerprint(source/f) for f in files},
        policy_source=str(grid.relative_to(ROOT)),policy_source_sha256=fingerprint(grid),
        encoded_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        encoded_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        test_indices_sha256=hashlib.sha256(test.tobytes()).hexdigest(),
        preprocessing='Reuse saved feature transformations, category order and exact train/test row indices; no refitting, filtering or new split.',
        validation='No test-based parameter or checkpoint selection.')
    out=OUT/dataset
    out.mkdir(parents=True,exist_ok=True)
    values=dict(x=x,y=y,train_indices=train,test_indices=test)
    path=out/'inputs.npz'
    if path.exists():
        with np.load(path) as old:
            for k,v in values.items(): np.testing.assert_array_equal(old[k],v)
        assert json.loads((out/'data_manifest.json').read_text())==metadata
    else:
        np.savez_compressed(path,**values)
        (out/'data_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ('dataset','rows','dimensions','train_rows','test_rows','policy','encoded_x_sha256')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['credit','oulad'])
    args=p.parse_args()
    for dataset in ([args.dataset] if args.dataset else ['credit','oulad']): prepare(dataset)
