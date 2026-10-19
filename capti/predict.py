"""Predict standalone assay probabilities using the final CAPTI-Net model."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .io import read_json, sha256
from .preprocessing import transform
from .model import CAPTI


def neural_predict(smiles, head_path, backbone_path=None, device='cpu', legacy_layout=None):
    # Load checkpoints only from trusted sources.
    obj = torch.load(head_path, map_location='cpu', weights_only=False)
    tasks = obj['task_names']
    uses_graph = obj['config'].get('base_scale', .5) != 0
    if not len(smiles):
        raise ValueError('No molecules supplied.')
    if uses_graph:
        if backbone_path is None:
            raise ValueError('This variant requires its matching graph checkpoint.')
        expected = obj.get('graph_checkpoint_sha256')
        if expected is not None and sha256(backbone_path) != expected:
            raise ValueError('Graph checkpoint differs from the checkpoint used to fit this head.')
    if legacy_layout is None:
        legacy_layout = obj.get('legacy_input_layout', False)
    from .features import ChemicalFeatures
    counts, topology = ChemicalFeatures().transform(smiles)
    x, c = transform(counts, topology, obj['stats'], legacy_layout)
    if uses_graph:
        from .backbone import extract_logits
        base = extract_logits(backbone_path, smiles, tasks, device)
    else:
        base = np.zeros((len(smiles), len(tasks)), np.float32)
    model = CAPTI(len(tasks), obj['config']).to(device)
    model.load_state_dict(obj['state'], strict=True)
    model.eval()
    with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.float16,
                                               enabled=str(device).startswith('cuda')):
        pred = model(torch.as_tensor(x, device=device), torch.as_tensor(c, device=device),
                     torch.as_tensor(base, device=device)).float().sigmoid().cpu().numpy()
    if pred.shape != (len(smiles), len(tasks)) or not np.isfinite(pred).all():
        raise ValueError('Invalid neural prediction array.')
    return pred, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--backbone', help='Required for variants retaining the graph pathway.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    molecules = read_json(args.molecules)
    smiles = molecules['smiles']
    pred, tasks = neural_predict(smiles, args.head, args.backbone, args.device)
    # p is the final output and is directly accepted by the evaluation CLI.
    result = {'p': pred, 'smiles': np.asarray(smiles), 'tasks': np.asarray(tasks)}
    if 'ids' in molecules:
        ids = np.asarray(molecules['ids'])
        if ids.shape != (len(smiles),) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
            raise ValueError('ids must be unique integer row identifiers, one per molecule.')
        result['ids'] = ids
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print('Saved', len(smiles), 'molecules x', len(tasks), 'endpoints to', out)


if __name__ == '__main__':
    main()
