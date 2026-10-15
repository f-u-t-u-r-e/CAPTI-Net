"""Checked file formats and row/task alignment."""
from pathlib import Path
import hashlib, json
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path

def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in ['count', 'topology', 'smiles']}
    n = len(d['smiles'])
    if n == 0 or d['count'].shape != (n,2048) or d['topology'].shape != (n,3,152):
        raise ValueError('Expected nonempty count[N,2048], topology[N,3,152], smiles[N].')
    for k in ['count', 'topology']:
        if not np.isfinite(d[k]).all() or np.any(d[k] < 0):
            raise ValueError(f'Invalid nonnegative feature array: {k}')
    return d

def load_split(path, n, tasks):
    # Deliberately load only these four members: no held-out labels.
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in ['train','val','y_train','y_val']}
    for part in ['train','val']:
        ids = s[part]
        y = s['y_'+part]
        if ids.ndim != 1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)==0:
            raise ValueError('Partition indices must be nonempty integer vectors.')
        if len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=n):
            raise ValueError('Duplicate or out-of-range partition index.')
        if y.shape != (len(ids),len(tasks)) or np.isinf(y).any():
            raise ValueError('Wrong label shape or infinite labels.')
        if not np.isin(y[np.isfinite(y)],[0,1]).all():
            raise ValueError('Observed labels must be 0 or 1; missing labels must be NaN.')
    if np.intersect1d(s['train'],s['val']).size:
        raise ValueError('Training and validation overlap.')
    return s

def aligned_array(path, key, smiles, tasks):
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['smiles'],smiles) or z['tasks'].tolist()!=list(tasks):
            raise ValueError(f'Row/task mismatch in {path}')
        a=z[key].astype(np.float32)
    if a.shape!=(len(smiles),len(tasks)) or not np.isfinite(a).all():
        raise ValueError(f'Invalid {key} array.')
    return a
