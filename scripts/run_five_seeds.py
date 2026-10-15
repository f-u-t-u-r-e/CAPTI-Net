"""Train the adopted CAPTI-Net with one common configuration on ten partitions."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    root=Path(a.data_root).resolve();out=Path(a.output_root).resolve()
    project=Path(__file__).resolve().parents[1]
    for dataset in ['tox21','toxcast']:
        for seed in [13,23,37,53,71]:
            d=root/dataset
            subprocess.run([sys.executable,'-m','capti.train_head','--features',str(d/'chemical_features.npz'),
                '--split',str(d/f'split_{seed}.npz'),'--base',str(d/f'graph_{seed}.npz'),
                '--molecules',str(d/'molecules.json'),'--seed',str(seed),'--device',a.device,
                '--output',str(out/dataset/str(seed))],cwd=project,check=True)

if __name__=='__main__': main()
