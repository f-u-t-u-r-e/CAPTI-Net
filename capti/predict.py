"""Predict new SMILES with a trained CAPTI-Net head and its matching graph checkpoint."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from .io import read_json, write_json
from .features import ChemicalFeatures
from .preprocessing import transform
from .model import CAPTI
from .backbone import extract_logits

def neural_predict(smiles, head_path, backbone_path, device='cpu', legacy_layout=None):
    # Load only checkpoints from trusted sources; PyTorch checkpoints are not a safe interchange format.
    obj=torch.load(head_path,map_location='cpu',weights_only=False)
    tasks=obj['task_names']
    if legacy_layout is None: legacy_layout=obj.get('legacy_input_layout',False)
    counts,topology=ChemicalFeatures().transform(smiles)
    x,c=transform(counts,topology,obj['stats'],legacy_layout)
    base=extract_logits(backbone_path,smiles,tasks,device)
    model=CAPTI(len(tasks),obj['config']).to(device)
    model.load_state_dict(obj['state'],strict=True);model.eval()
    with torch.inference_mode(),torch.autocast(torch.device(device).type,dtype=torch.float16,
                                             enabled=str(device).startswith('cuda')):
        p=model(torch.as_tensor(x,device=device),torch.as_tensor(c,device=device),
                torch.as_tensor(base,device=device)).float().sigmoid().cpu().numpy()
    return p,tasks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecules',required=True)
    p.add_argument('--head',required=True)
    p.add_argument('--backbone',required=True)
    p.add_argument('--partition',help='Backbone partition.json; required for RF fusion.')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--rf-dir',help='Optional fitted RF folder; adds fixed 50:50 probability fusion.')
    p.add_argument('--rf-python',help='Python executable from the RDKit 2026.09.1 environment.')
    p.add_argument('--dataset',choices=['tox21','toxcast'],help='Required with --rf-dir.')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if a.threads<1: p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    if out.exists(): raise FileExistsError(out)
    if a.rf_dir and (not a.rf_python or not a.dataset or not a.partition):
        p.error('--rf-dir requires --rf-python, --dataset and --partition')
    m=read_json(a.molecules);smiles=m['smiles']
    # Preserve row order and input spelling. No silent invalid-row removal.
    pred,tasks=neural_predict(smiles,a.head,a.backbone,a.device)
    result={'p_neural':pred,'smiles':np.asarray(smiles),'tasks':np.asarray(tasks)}
    if a.rf_dir:
        from .forest import predict as predict_rf
        meta=read_json(Path(a.rf_dir)/'metadata.json')
        head=torch.load(a.head,map_location='cpu',weights_only=False)
        partition=read_json(a.partition)
        if meta['seed']!=head['seed'] or meta['train']!=partition['train']:
            raise ValueError('RF and neural models must use the same training partition.')
        root=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);write_json(temp/'molecules.json',{'smiles':smiles})
            subprocess.run([a.rf_python,'-m','capti.rf_features','--molecules',str(temp/'molecules.json'),
                '--dataset',a.dataset,'--descriptor-names',str(root/'assets/descriptor_names.json'),
                '--output',str(temp/'rf.npz')],cwd=root,check=True)
            with np.load(temp/'rf.npz',allow_pickle=False) as z:
                rf=predict_rf(a.rf_dir,z['fp'],z['desc'],tasks,z['descriptor_names'].tolist())
        result.update(p_rf=rf,p_fused=.5*pred+.5*rf)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**result)
    print('Saved',len(smiles),'molecules x',len(tasks),'endpoints to',out)

if __name__=='__main__': main()
