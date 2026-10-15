"""Fit statistics only on the training partition."""
import numpy as np

def fit_topology(raw, train_ids):
    v=np.log1p(np.asarray(raw,dtype=np.float32))
    return {'pharmacophore_mean':v[train_ids].mean(0),
            'pharmacophore_std':np.maximum(v[train_ids].std(0),.2)}

def transform(count, topology, stats, legacy_layout=False):
    # Casting before log1p is intentional: uint16 input has different ufunc promotion.
    x=np.log1p(np.asarray(count,dtype=np.float32))
    c=np.log1p(np.asarray(topology,dtype=np.float32))
    c=np.clip((c-stats['pharmacophore_mean'])/stats['pharmacophore_std'],-10,10).astype(np.float32)
    if legacy_layout:
        # Unused zero columns retain the archived FP16 matrix stride, not extra features.
        x=np.pad(x,((0,0),(0,3417-2048)))
    return x,c
