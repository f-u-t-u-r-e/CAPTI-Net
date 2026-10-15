"""Two chemical views; feature generation does not read labels."""
import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import ChemicalFeatures, rdFingerprintGenerator
from .io import read_json, write_json, sha256

FAMILIES = ['Donor','Acceptor','NegIonizable','PosIonizable','ZnBinder','Aromatic','Hydrophobe','LumpedHydrophobe']
PAIRS = [(a,b) for a in range(8) for b in range(a,8)]
PAIR_INDEX = {p:i for i,p in enumerate(PAIRS)}
FDEF = Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'
EXPECTED_FDEF = '87ab54bd717af50a78a710d23e7737a2cafc96e5293bf6257ecdda86b6da3f65'

def check_environment():
    if rdBase.rdkitVersion != '2024.09.6':
        raise RuntimeError('Chemical branches require RDKit 2024.09.6. Use the core environment.')
    if sha256(FDEF) != EXPECTED_FDEF:
        raise RuntimeError('BaseFeatures.fdef mismatch; check RDBASE and RDKit installation.')

class ChemicalFeatures:
    def __init__(self):
        check_environment()
        from rdkit.Chem import ChemicalFeatures as CF
        self.factory = CF.BuildFeatureFactory(str(FDEF))
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)

    def one(self, smiles):
        mol=Chem.MolFromSmiles(str(smiles))
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f'Invalid or empty SMILES: {smiles!r}')
        features=[f for f in self.factory.GetFeaturesForMol(mol) if f.GetFamily() in FAMILIES]
        masks=[0]*mol.GetNumAtoms()
        for f in features:
            bit=1<<FAMILIES.index(f.GetFamily())
            for a in f.GetAtomIds(): masks[a]|=bit
        counts=self.generator.GetCountFingerprintAsNumPy(mol,customAtomInvariants=[m+1 for m in masks])
        labels=np.array([FAMILIES.index(f.GetFamily()) for f in features],dtype=np.int64)
        atoms=[np.array(f.GetAtomIds(),dtype=np.int64) for f in features]
        result=np.zeros((3,152),np.float32)
        result[:,:8]=np.bincount(labels,minlength=8).astype(np.float32)
        distances=Chem.GetDistanceMatrix(mol)
        for i in range(len(features)):
            for j in range(i+1,len(features)):
                d=float(distances[np.ix_(atoms[i],atoms[j])].min())
                if d>=1e7: continue
                bucket=min(int(d),11)
                key=tuple(sorted([int(labels[i]),int(labels[j])]))
                result[bucket//4,8+PAIR_INDEX[key]*4+bucket%4]+=1
        if counts.max()>65535: raise ValueError('Count exceeds uint16 cache capacity.')
        return counts.astype(np.uint16),result

    def transform(self, smiles):
        if not len(smiles): raise ValueError('No molecules supplied.')
        rows=[self.one(s) for s in smiles]
        return np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecules',required=True,help='JSON object with a smiles list.')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    smiles=read_json(a.molecules)['smiles']
    count,topology=ChemicalFeatures().transform(smiles)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,count=count,topology=topology,smiles=np.asarray(smiles))
    write_json(str(out)+'.json',{'rdkit':rdBase.rdkitVersion,'fdef_sha256':sha256(FDEF),
        'molecules_sha256':sha256(a.molecules),'cache_sha256':sha256(out),'rows':len(smiles)})

if __name__=='__main__': main()
