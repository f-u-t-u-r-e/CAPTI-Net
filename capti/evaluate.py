"""Score an explicitly supplied train/validation prediction file."""
import argparse
from pathlib import Path
import numpy as np
from .io import write_json
from .metrics import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',required=True)
    p.add_argument('--part',choices=['train','val'],default='val')
    p.add_argument('--predictions',required=True,help='NPZ with ids and p in exactly that partition order.')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError(a.output)
    with np.load(a.split,allow_pickle=False) as s,np.load(a.predictions,allow_pickle=False) as z:
        if not np.array_equal(s[a.part],z['ids']): raise ValueError('Prediction row order mismatch.')
        y=s['y_'+a.part];pred=z['p']
        if y.shape!=pred.shape or not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):
            raise ValueError('Invalid probability array.')
        report=metrics(y,pred)
    write_json(a.output,report)

if __name__=='__main__': main()
