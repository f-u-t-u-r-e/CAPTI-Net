"""Chemprop 1.6.1 graph backbone: official optimizer, loss and Noam scheduler."""
import argparse
import copy
import logging
import random
from pathlib import Path
import numpy as np
import torch
from .io import read_json, write_json, fresh_dir, load_split, sha256
from .metrics import metrics

def make_loader(smiles, tasks, labels=None, shuffle=False, seed=0):
    from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
    if labels is None:
        labels = np.full((len(smiles), len(tasks)), np.nan)
    rows = [MoleculeDatapoint(smiles=[s],targets=[None if np.isnan(v) else float(v) for v in y])
            for s,y in zip(smiles,labels)]
    return MoleculeDataLoader(MoleculeDataset(rows), batch_size=128, num_workers=0, shuffle=shuffle, seed=seed)

def extract_logits(checkpoint, smiles, tasks, device='cpu'):
    from chemprop.utils import load_checkpoint, load_args
    from chemprop.train import predict
    model = load_checkpoint(str(checkpoint), device=torch.device(device))
    if list(load_args(str(checkpoint)).task_names) != list(tasks):
        raise ValueError('Backbone task order does not match the head.')
    values = []
    handle = model.readout[-1].register_forward_hook(lambda module, args, out: values.append(out.detach().float().cpu()))
    try:
        with torch.inference_mode():
            predict(model, make_loader(smiles,tasks), disable_progress_bar=True)
    finally:
        handle.remove()
    logits = torch.cat(values).numpy()
    if logits.shape != (len(smiles),len(tasks)) or not np.isfinite(logits).all():
        raise ValueError('Invalid graph logits.')
    return logits

def fit(molecules, split_path, seed, output, device='cuda'):
    import pandas as pd
    from chemprop.args import TrainArgs
    from chemprop.models import MoleculeModel
    from chemprop.train import train, predict
    from chemprop.train.loss_functions import get_loss_func
    from chemprop.utils import build_optimizer, build_lr_scheduler, save_checkpoint
    torch.set_num_threads(6)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith('cuda'): torch.cuda.manual_seed_all(seed)
    m = read_json(molecules); smiles, tasks = m['smiles'], m['tasks']
    s = load_split(split_path,len(smiles),tasks)
    out = fresh_dir(output)
    frame = pd.DataFrame(s['y_train'],columns=tasks)
    frame.insert(0,'smiles',[smiles[i] for i in s['train']])
    frame.to_csv(out/'train_only.csv',index=False)
    arguments = ['--data_path',str(out/'train_only.csv'),'--dataset_type','classification',
        '--metric','prc-auc','--save_dir',str(out),'--epochs','80','--batch_size','128',
        '--hidden_size','300','--depth','3','--dropout','0','--max_lr','0.001',
        '--num_workers','0','--seed',str(seed),'--pytorch_seed',str(seed),'--quiet']
    arguments += ['--gpu', str(torch.device(device).index or 0)] if str(device).startswith('cuda') else ['--no_cuda']
    args = TrainArgs().parse_args(arguments)
    args.task_names=tasks; args.train_data_size=len(s['train']); args.features_size=0
    args.target_weights=[1.]*len(tasks)
    tr=make_loader([smiles[i] for i in s['train']],tasks,s['y_train'],True,seed)
    va=make_loader([smiles[i] for i in s['val']],tasks,s['y_val'])
    if metrics(s['y_val'],np.zeros_like(s['y_val']))['macro_ap'] is None:
        raise ValueError('No validation endpoint has two classes.')
    model=MoleculeModel(args).to(args.device)
    optimizer=build_optimizer(model,args); scheduler=build_lr_scheduler(optimizer,args)
    loss_func=get_loss_func(args); best=-1.; best_ep=0; n_iter=0; history=[]
    logger=logging.getLogger('capti.backbone'); logger.addHandler(logging.NullHandler()); logger.propagate=False
    for epoch in range(1,81):
        n_iter=train(model,tr,loss_func,optimizer,scheduler,args,n_iter=n_iter,logger=logger)
        pred=np.asarray(predict(model,va,disable_progress_bar=True),np.float32)
        report=metrics(s['y_val'],pred)
        history.append({'epoch':epoch,'macro_ap':report['macro_ap'],'macro_auroc':report['macro_auroc']})
        if report['macro_ap']>best+1e-7:
            best=report['macro_ap']; best_ep=epoch; saved=copy.deepcopy(model.state_dict())
        if epoch-best_ep>=25: break
    model.load_state_dict(saved)
    save_checkpoint(str(out/'backbone.pt'),model,args=args)
    write_json(out/'training.json',{'seed':seed,'best_epoch':best_ep,'history':history,
        'split_sha256':sha256(split_path),'backbone_sha256':sha256(out/'backbone.pt')})
    write_json(out/'partition.json',{'train':s['train'].tolist(),'seed':seed,'tasks':tasks})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--split',required=True); tr.add_argument('--seed',type=int,required=True)
    ca=sub.add_parser('cache'); ca.add_argument('--checkpoint',required=True)
    ca.add_argument('--partition',required=True,help='JSON with actual backbone training indices (train).')
    for q in [tr,ca]:
        q.add_argument('--molecules',required=True); q.add_argument('--output',required=True)
        q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.command=='train': fit(a.molecules,a.split,a.seed,a.output,a.device)
    else:
        out=Path(a.output)
        if out.exists(): raise FileExistsError(out)
        m=read_json(a.molecules); partition=read_json(a.partition)
        if partition['tasks']!=m['tasks']: raise ValueError('Partition task order mismatch.')
        v=extract_logits(a.checkpoint,m['smiles'],m['tasks'],a.device)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,logits=v,smiles=np.asarray(m['smiles']),tasks=np.asarray(m['tasks']),
                            train_ids=np.asarray(partition['train'],np.int64))
        write_json(str(out)+'.json',{'checkpoint_sha256':sha256(a.checkpoint),'cache_sha256':sha256(out)})

if __name__=='__main__': main()
