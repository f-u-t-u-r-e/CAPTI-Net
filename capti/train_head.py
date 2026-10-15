"""Fit the two-branch correction with a frozen, partition-matched graph base."""
import argparse
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .io import read_json, write_json, fresh_dir, load_features, load_split, aligned_array, sha256
from .model import CAPTI, masked_loss
from .preprocessing import fit_topology, transform
from .metrics import metrics

def fit(features, split_path, base_path, tasks, config, seed, output,
        device='cuda', legacy_layout=False):
    config = dict(config)
    if config['epochs'] < 1 or config['patience'] < 1:
        raise ValueError('Epoch and patience counts must be positive.')
    device = torch.device(device)
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    f = load_features(features)
    s = load_split(split_path, len(f['smiles']), tasks)
    base_np = aligned_array(base_path, 'logits', f['smiles'], tasks)
    # Reject a cache explicitly marked as coming from another training partition.
    with np.load(base_path, allow_pickle=False) as z:
        if 'train_ids' not in z or not np.array_equal(z['train_ids'], s['train']):
            raise ValueError('Graph cache must identify exactly the same training indices.')
    if not np.isfinite(s['y_train']).any():
        raise ValueError('No observed training labels.')
    stats = fit_topology(f['topology'], s['train'])
    xn, cn = transform(f['count'], f['topology'], stats, legacy_layout)
    x = torch.as_tensor(xn, device=device)
    c = torch.as_tensor(cn, device=device)
    base = torch.as_tensor(base_np, device=device)
    tr, va = s['train'], s['val']
    y = torch.as_tensor(np.nan_to_num(s['y_train'], nan=0).astype(np.float32), device=device)
    mask = torch.as_tensor(np.isfinite(s['y_train']), device=device)
    model = CAPTI(len(tasks), config).to(device)
    with torch.no_grad():
        model.fragment_mean.copy_(F.normalize(x[tr, :2048], p=2, dim=1).mean(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    def prediction():
        model.eval()
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            return model(x[va], c[va], base[va]).float().sigmoid().cpu().numpy()

    initial = metrics(s['y_val'], prediction())
    if initial['macro_ap'] is None:
        raise ValueError('Validation has no endpoint with both observed classes.')
    out = fresh_dir(output)
    started = time.perf_counter()
    best, best_epoch = initial['macro_ap'], 0
    saved = copy.deepcopy(model.state_dict())
    history = [{'epoch': 0, 'macro_ap': best, 'macro_auroc': initial['macro_auroc']}]
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            loss = masked_loss(model(x[tr], c[tr], base[tr]), y, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at epoch {epoch}.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer)
        scaler.update()
        if epoch % 5 == 0:
            report = metrics(s['y_val'], prediction())
            history.append({'epoch': epoch, 'loss': float(loss.detach()),
                            'macro_ap': report['macro_ap'], 'macro_auroc': report['macro_auroc']})
            if report['macro_ap'] > best + 1e-7:
                best, best_epoch = report['macro_ap'], epoch
                saved = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= config['patience']:
                break
    model.load_state_dict(saved)
    p = prediction()
    obj = {'state': {k: v.detach().cpu() for k, v in saved.items()},
           'config': config, 'stats': stats, 'tasks': len(tasks), 'task_names': list(tasks),
           'seed': seed, 'legacy_input_layout': legacy_layout}
    torch.save(obj, out / 'head.pt')
    np.savez_compressed(out / 'validation_predictions.npz', ids=va, p=p,
                        smiles=f['smiles'][va], tasks=np.asarray(tasks))
    report = {'seed': seed, 'config': config, 'best_epoch': best_epoch,
              'epochs_run': epoch, 'validation': metrics(s['y_val'], p), 'history': history,
              'seconds': time.perf_counter() - started,
              'registered_head_parameters': sum(v.numel() for v in model.parameters()),
              'input_sha256': {k: sha256(v) for k, v in [('features',features),('split',split_path),('base',base_path)]},
              'head_sha256': sha256(out/'head.pt')}
    write_json(out / 'training.json', report)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--molecules', required=True, help='JSON containing ordered task names.')
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    molecules = read_json(a.molecules)
    if load_features(a.features)['smiles'].tolist() != molecules['smiles']:
        raise ValueError('Chemical feature rows differ from molecules.json.')
    r = fit(a.features,a.split,a.base,molecules['tasks'],read_json(a.config),
            a.seed,a.output,a.device)
    print('Saved:', a.output, 'best epoch:', r['best_epoch'])

if __name__ == '__main__':
    main()
