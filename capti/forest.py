"""The fixed 400-tree per-endpoint random forest used in equal-weight fusion."""
import argparse
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .io import read_json, write_json, load_split, fresh_dir

def fit_stats(desc, ids):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median=np.nanmedian(desc[ids],axis=0)
    median=np.nan_to_num(median,nan=0.)
    filled=np.where(np.isfinite(desc[ids]),desc[ids],median)
    return {'median':median.astype(np.float32),'mean':filled.mean(0).astype(np.float32),
            'std':np.maximum(filled.std(0),1e-5).astype(np.float32)}

def transform(fp,desc,stats):
    filled=np.where(np.isfinite(desc),desc,stats['median'])
    filled=np.clip((filled-stats['mean'])/stats['std'],-10,10)
    return np.concatenate([fp.astype(np.float32),filled],axis=1).astype(np.float32)

def predict(folder, fp, desc, task_names, descriptor_names):
    folder=Path(folder);meta=read_json(folder/'metadata.json')
    if meta['tasks']!=list(task_names) or meta['descriptor_names']!=list(descriptor_names):
        raise ValueError('RF task/descriptor order mismatch.')
    x=transform(fp,desc,joblib.load(folder/'rf_stats.joblib'))
    p=np.empty((len(x),len(task_names)),np.float32)
    for t in range(len(task_names)):
        m=joblib.load(folder/f'rf_{t:03d}.joblib')
        p[:,t]=m if isinstance(m,float) else m.predict_proba(x)[:,1]
    return p

def fit(features,split_path,tasks,seed,output):
    with np.load(features,allow_pickle=False) as z:
        fp,desc=z['fp'],z['desc'];smiles=z['smiles'];names=z['descriptor_names'].tolist()
    if fp.shape!=(len(smiles),2048) or desc.shape!=(len(smiles),217):
        raise ValueError('Expected binary Morgan2048 and 217 descriptors.')
    split=load_split(split_path,len(fp),tasks);tr=split['train'];y=split['y_train']
    stats=fit_stats(desc,tr);x=transform(fp,desc,stats);out=fresh_dir(output)
    for t in range(len(tasks)):
        obs=np.isfinite(y[:,t]);labels=y[obs,t]
        if not len(labels) or np.unique(labels).size<2:
            model=float((labels.sum()+1)/(len(labels)+2))
        else:
            model=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features='sqrt',
                class_weight='balanced_subsample',n_jobs=1,random_state=seed)
            model.fit(x[tr[obs]],labels.astype(np.int64))
        joblib.dump(model,out/f'rf_{t:03d}.joblib',compress=3)
    joblib.dump(stats,out/'rf_stats.joblib',compress=3)
    write_json(out/'metadata.json',{'tasks':list(tasks),'descriptor_names':names,'seed':seed,
        'train':tr.tolist(),'config':{'trees':400,'leaf':2,'descriptors':True}})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('train');a.add_argument('--split',required=True);a.add_argument('--seed',type=int,required=True)
    b=sub.add_parser('predict');b.add_argument('--model-dir',required=True)
    for q in [a,b]:
        q.add_argument('--features',required=True);q.add_argument('--molecules',required=True);q.add_argument('--output',required=True)
    v=p.parse_args();m=read_json(v.molecules)
    with np.load(v.features,allow_pickle=False) as z:
        if z['smiles'].tolist()!=m['smiles']: raise ValueError('Molecule row mismatch.')
    if v.command=='train': fit(v.features,v.split,m['tasks'],v.seed,v.output)
    else:
        if Path(v.output).exists(): raise FileExistsError(v.output)
        with np.load(v.features,allow_pickle=False) as z:
            if z['smiles'].tolist()!=m['smiles']: raise ValueError('Molecule row mismatch.')
            p=predict(v.model_dir,z['fp'],z['desc'],m['tasks'],z['descriptor_names'].tolist())
        np.savez_compressed(v.output,p=p,smiles=np.asarray(m['smiles']),tasks=np.asarray(m['tasks']))

if __name__=='__main__': main()
