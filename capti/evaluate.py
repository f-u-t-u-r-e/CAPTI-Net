"""Score explicitly supplied standalone predictions against aligned observed labels."""
import argparse
from pathlib import Path
import numpy as np
from .io import write_json
from .metrics import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--split',help='Development split containing train/val labels.')
    source.add_argument('--labels',help='Explicit evaluation NPZ containing ids, y and tasks.')
    p.add_argument('--part',choices=['train','val'],default='val')
    p.add_argument('--predictions',required=True,help='NPZ with ids and p in exactly that partition order.')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    with np.load(a.labels or a.split,allow_pickle=False) as s,np.load(a.predictions,allow_pickle=False) as z:
        ids=s['ids'] if a.labels else s[a.part]
        if not np.array_equal(ids,z['ids']): raise ValueError('Prediction row order mismatch.')
        if a.labels and not np.array_equal(s['tasks'],z['tasks']): raise ValueError('Prediction task order mismatch.')
        y=s['y'] if a.labels else s['y_'+a.part];pred=z['p']
        if y.shape!=pred.shape or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):
            raise ValueError('Invalid probability array.')
        report=metrics(y,pred)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,report)

if __name__=='__main__': main()
