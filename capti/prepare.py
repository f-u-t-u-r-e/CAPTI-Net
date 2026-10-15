"""Public benchmark cleaning and fixed scaffold partitions (RDKit 2026.09.1)."""
import os
os.environ.pop('RDBASE',None)
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from .io import read_json,write_json,fresh_dir,sha256

def clean(raw, dataset):
    frame=pd.read_csv(raw)
    tasks=([c for c in frame if c.startswith(('NR-','SR-'))] if dataset=='tox21'
           else [c for c in frame if c!='smiles'])
    grouped=defaultdict(list);invalid=0
    for _,row in frame.iterrows():
        mol=Chem.MolFromSmiles(str(row['smiles']))
        if mol is None:invalid+=1;continue
        smi=Chem.MolToSmiles(mol,isomericSmiles=True)
        grouped[smi].append(row[tasks].to_numpy(dtype=np.float32))
    smiles=[];labels=[];scaffolds=[];mols=[];conflicts=0
    for smi,values in grouped.items():
        values=np.stack(values);out=np.full(len(tasks),np.nan,np.float32)
        for t in range(len(tasks)):
            seen=np.unique(values[:,t][np.isfinite(values[:,t])])
            if len(seen)==1:out[t]=seen[0]
            elif len(seen)>1:conflicts+=1
        mol=Chem.MolFromSmiles(smi)
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False) or 'ACYCLIC'
        smiles.append(smi);labels.append(out);scaffolds.append(scaffold);mols.append(mol)
    return smiles,np.stack(labels),scaffolds,mols,tasks,dict(raw_rows=len(frame),invalid=invalid,
            unique_molecules=len(smiles),conflicting_cells=conflicts,task_count=len(tasks))

def split_scaffolds(scaffolds,seed):
    groups=defaultdict(list)
    for i,s in enumerate(scaffolds):groups[s].append(i)
    rng=np.random.default_rng(seed);keys=list(groups);rng.shuffle(keys)
    n=len(scaffolds);caps=np.array([int(.7*n),int(.15*n),n-int(.7*n)-int(.15*n)])
    # Oversized groups cannot fit either holdout fold. Assign them to training.
    # Other groups remain randomized; sorting all groups would leave almost no
    # within-test MMPs and would barely change test sets across seeds.
    train_keys=[k for k in keys if len(groups[k])>min(caps[1:])]
    used=sum(len(groups[k]) for k in train_keys);holdout=[]
    for key in keys:
        if key in train_keys:continue
        if used+len(groups[key])<=caps[0]:train_keys.append(key);used+=len(groups[key])
        else:holdout.append(key)
    # Label-blind subset-sum allocation reaches the validation target exactly
    # when possible, without breaking scaffold groups or cherry-picking labels.
    rng.shuffle(holdout);target=int(caps[1]);reachable=np.zeros(target+1,bool);reachable[0]=True
    parent_group=np.full(target+1,-1,int);parent_sum=np.full(target+1,-1,int)
    for j,key in enumerate(holdout):
        size=len(groups[key])
        if size>target:continue
        new=np.flatnonzero(reachable[:target-size+1]&~reachable[size:])+size
        parent_group[new]=j;parent_sum[new]=new-size;reachable[new]=True
        if reachable[target]:break
    cursor=int(np.flatnonzero(reachable)[-1]);val_keys=set()
    while cursor>0:
        j=int(parent_group[cursor]);val_keys.add(holdout[j]);cursor=int(parent_sum[cursor])
    parts=[[i for k in train_keys for i in groups[k]],
           [i for k in holdout if k in val_keys for i in groups[k]],
           [i for k in holdout if k not in val_keys for i in groups[k]]]
    sets=[set(x) for x in parts]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert len(set.union(*sets))==n
    ssets=[{scaffolds[i] for i in ids} for ids in parts]
    assert not (ssets[0]&ssets[1] or ssets[0]&ssets[2] or ssets[1]&ssets[2])
    return [np.array(sorted(x),np.int64) for x in parts]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True,help='Public tox21.csv.gz or toxcast_data.csv.gz.')
    p.add_argument('--dataset',choices=['tox21','toxcast'],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if rdBase.rdkitVersion!='2026.09.1': raise RuntimeError('Use the preparation/RF environment with RDKit 2026.09.1.')
    smiles,y,scaffolds,_,tasks,meta=clean(a.raw,a.dataset)
    assets=Path(__file__).resolve().parent.parent/'assets'
    reference=read_json(assets/f'{a.dataset}_molecules.json')
    if smiles!=reference['smiles'] or tasks!=reference['tasks']:
        raise ValueError('Public data release/cleaning differs from the packaged molecule or task order.')
    if not np.isin(y[np.isfinite(y)],[0,1]).all(): raise ValueError('Non-binary labels.')
    out=fresh_dir(a.output)
    write_json(out/'molecules.json',{'smiles':smiles,'tasks':tasks,'scaffold':scaffolds})
    fixed=read_json(assets/'partition_indices.json')['partitions'][a.dataset]
    for seed in [13,23,37,53,71]:
        tr,va,te=split_scaffolds(scaffolds,seed)
        for name,ids in [('train',tr),('val',va),('test',te)]:
            if ids.tolist()!=fixed[str(seed)][name]: raise ValueError('Scaffold partition differs from archived indices.')
        # This release emits development labels only. No held-out label file is created.
        np.savez_compressed(out/f'split_{seed}.npz',train=tr,val=va,test=te,y_train=y[tr],y_val=y[va])
    meta.update(raw_sha256=sha256(a.raw),rdkit=rdBase.rdkitVersion)
    write_json(out/'preparation.json',meta)
    print('Prepared',a.dataset,meta['unique_molecules'],'molecules.')

if __name__=='__main__': main()
