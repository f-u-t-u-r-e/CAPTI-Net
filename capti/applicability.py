"""Nearest-training Tanimoto bins with shared, observed-label assay eligibility."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .io import read_json, write_json, fresh_dir
from .metrics import metrics

BINS = ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']


def bin_indices(similarity):
    s = np.asarray(similarity, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Similarities must be a vector in [0,1].')
    # 0.3 enters bin 2, 0.5 enters bin 3, and 0.7 remains in bin 3.
    return np.where(s < .3, 0, np.where(s < .5, 1, np.where(s <= .7, 2, 3)))


def nearest_training(smiles, train_ids, query_ids):
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    if rdBase.rdkitVersion != '2024.09.6':
        raise RuntimeError('Use RDKit 2024.09.6 for the locked similarity definition.')
    arrays = [np.asarray(v) for v in [train_ids, query_ids]]
    for ids in arrays:
        if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Training and query indices must be nonempty integer vectors.')
        if len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(smiles))):
            raise ValueError('Duplicate or out-of-range molecule index.')
    train_ids, query_ids = arrays
    if np.intersect1d(train_ids, query_ids).size:
        raise ValueError('Query molecules must be disjoint from supervised training molecules.')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = {}
    for i in np.r_[train_ids, query_ids]:
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f'Invalid molecule at row {i}.')
        fingerprints[int(i)] = generator.GetFingerprint(mol)
    reference = [fingerprints[int(i)] for i in train_ids]
    similarities, nearest = [], []
    for i in query_ids:
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[int(i)], reference))
        index = int(np.argmax(values))
        similarities.append(float(values[index])); nearest.append(int(train_ids[index]))
    return np.asarray(similarities), np.asarray(nearest)


def score_bins(y, predictions, similarity):
    """One matched partition; all methods share each bin's eligible assay set."""
    y = np.asarray(y, dtype=np.float64)
    bins = bin_indices(similarity)
    if y.ndim != 2 or len(y) != len(bins) or not predictions:
        raise ValueError('Labels, similarities and at least one prediction matrix are required.')
    for pred in predictions.values():
        metrics(y, pred)  # Validate without changing any labels or probabilities.
    masks = [bins == i for i in range(4)]
    eligible = [np.any(y[m] == 0, axis=0) & np.any(y[m] == 1, axis=0) for m in masks]
    common = np.logical_and.reduce(eligible)
    rows, per_task = [], []
    for mode in ['bin_specific', 'common_across_bins']:
        for index, mask in enumerate(masks):
            ys = y[mask].copy()
            allowed = eligible[index] if mode == 'bin_specific' else common
            ys[:, ~allowed] = np.nan
            for method, pred in predictions.items():
                report = metrics(ys, np.asarray(pred)[mask])
                prevalence = [v['positives'] / v['n'] for v in report['per_task']]
                rows.append({'eligibility': mode, 'bin': BINS[index], 'method': method,
                             'molecules': int(mask.sum()), 'eligible_tasks': report['eligible_tasks'],
                             'macro_prevalence': float(np.mean(prevalence)) if prevalence else None,
                             'macro_ap': report['macro_ap'], 'macro_auroc': report['macro_auroc']})
                for task in report['per_task']:
                    per_task.append(dict(eligibility=mode, bin=BINS[index], method=method,
                                         prevalence=task['positives']/task['n'], **task))
    return rows, per_task


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules', required=True, help='Full ordered molecules.json.')
    parser.add_argument('--split', required=True, help='NPZ containing training indices (train).')
    parser.add_argument('--labels', required=True, help='Explicit held-out NPZ: ids, y, tasks.')
    parser.add_argument('--predictions', action='append', required=True, metavar='METHOD=NPZ')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    molecules = read_json(args.molecules)
    with np.load(args.split, allow_pickle=False) as z:
        train = z['train']
    with np.load(args.labels, allow_pickle=False) as z:
        ids, y, tasks = z['ids'], z['y'], z['tasks']
    if 'tasks' in molecules and list(tasks) != molecules['tasks']:
        raise ValueError('Evaluation task order differs from molecules.json.')
    predictions = {}
    for spec in args.predictions:
        if '=' not in spec:
            parser.error('--predictions must be METHOD=NPZ')
        method, path = spec.split('=', 1)
        if not method or method in predictions:
            parser.error('Method names must be nonempty and unique.')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['ids'], ids) or not np.array_equal(z['tasks'], tasks):
                raise ValueError(f'Row/task alignment differs for {method}.')
            if 'smiles' in z and z['smiles'].tolist() != [molecules['smiles'][i] for i in ids]:
                raise ValueError(f'Molecule order differs for {method}.')
            predictions[method] = z['p']
    similarity, nearest = nearest_training(molecules['smiles'], train, ids)
    rows, per_task = score_bins(y, predictions, similarity)
    out = fresh_dir(args.output)
    write_csv(out/'scores.csv', rows, ['eligibility','bin','method','molecules','eligible_tasks',
                                      'macro_prevalence','macro_ap','macro_auroc'])
    write_csv(out/'task_scores.csv', per_task, ['eligibility','bin','method','prevalence','task','ap','auroc','n','positives'])
    write_csv(out/'domains.csv', [dict(molecule_index=int(i), nearest_training_index=int(j),
              max_training_tanimoto=float(s), bin=BINS[b]) for i,j,s,b in zip(ids,nearest,similarity,bin_indices(similarity))],
              ['molecule_index','nearest_training_index','max_training_tanimoto','bin'])
    write_json(out/'protocol.json', {'fingerprint':'Morgan, radius 2, 2048 binary bits, no chirality',
                                    'training_reference_only':True,'methods':list(predictions),
                                    'bins':['s<0.3','0.3<=s<0.5','0.5<=s<=0.7','s>0.7'],
                                    'eligibility':['bin_specific','common_across_bins'],
                                    'model_selection':False})


if __name__ == '__main__':
    main()
