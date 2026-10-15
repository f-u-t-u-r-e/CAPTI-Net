"""Binary Morgan + 217 descriptors, in the separate RDKit 2026.09.1 environment."""
import os
# Must happen before RDKit import: a parent RDKit process can export its Data path.
os.environ.pop('RDBASE',None)
import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem, rdBase, RDConfig
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from .io import read_json, sha256, write_json

def compute(smiles, names):
    if rdBase.rdkitVersion!='2026.09.1':
        raise RuntimeError('RF descriptor parity requires the separate RDKit 2026.09.1 environment.')
    functions=dict(Descriptors._descList)
    if len(names)!=217 or any(n not in functions for n in names):
        raise ValueError('The archived 217 descriptor names are required.')
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fps=[]; descriptors=[]
    for i,s in enumerate(smiles):
        mol=Chem.MolFromSmiles(s)
        if mol is None or not mol.GetNumAtoms(): raise ValueError(f'Invalid molecule at row {i}.')
        fps.append(generator.GetFingerprintAsNumPy(mol))
        values=[]
        for name in names:
            try: v=float(functions[name](mol))
            except Exception: v=np.nan
            values.append(v if np.isfinite(v) and abs(v)<1e15 else np.nan)
        descriptors.append(values)
    return np.asarray(fps,np.uint8),np.asarray(descriptors,np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecules',required=True);p.add_argument('--dataset',choices=['tox21','toxcast'],required=True)
    p.add_argument('--descriptor-names',default='assets/descriptor_names.json');p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    m=read_json(a.molecules);names=read_json(a.descriptor_names)[a.dataset]
    fp,desc=compute(m['smiles'],names);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,fp=fp,desc=desc,smiles=np.asarray(m['smiles']),descriptor_names=np.asarray(names))
    write_json(str(out)+'.json',{'rdkit':rdBase.rdkitVersion,'rows':len(fp),
                              'descriptor_names_sha256':sha256(a.descriptor_names),'cache_sha256':sha256(out)})

if __name__=='__main__': main()
