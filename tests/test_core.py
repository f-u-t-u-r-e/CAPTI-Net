"""Small synthetic checks. These are software checks, not benchmark evaluations."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from capti.features import ChemicalFeatures
from capti.model import CAPTI, masked_loss
from capti.preprocessing import fit_topology, transform
from capti.metrics import metrics
from capti.io import load_split, aligned_array
from capti.train_head import fit

ROOT=Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    def test_chemical_invariances(self):
        f=ChemicalFeatures()
        for left,right in [('CCO','OCC'),('Oc1ccccc1','c1ccccc1O'),('CC(=O)O','OC(C)=O')]:
            for a,b in zip(f.one(left),f.one(right)):
                np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(f.one('CCO.CCO')[1],2*f.one('CCO')[1])
        with self.assertRaises(ValueError):f.one('not a molecule')

    def test_feature_factory_matches_official_definition(self):
        from rdkit import Chem
        from rdkit.Chem import ChemicalFeatures as CF
        from capti.features import FDEF
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'features.fdef';path.write_bytes(FDEF.read_bytes())
            official=CF.BuildFeatureFactory(str(path));current=ChemicalFeatures().factory
            for smiles in ['CCO','Oc1ccccc1','CC(=O)O','C[NH3+]','CCO.CCO']:
                mol=Chem.MolFromSmiles(smiles)
                def labels(factory):
                    return [(v.GetFamily(),v.GetType(),v.GetAtomIds()) for v in factory.GetFeaturesForMol(mol)]
                self.assertEqual(labels(current),labels(official))

    def test_training_statistics_and_dtype(self):
        rng=np.random.default_rng(7);a=rng.integers(0,20,(8,3,152)).astype(np.float32)
        b=a.copy();b[4:]+=10000
        sa=fit_topology(a,np.arange(4));sb=fit_topology(b,np.arange(4))
        for k in sa:np.testing.assert_array_equal(sa[k],sb[k])
        counts=np.full((8,2048),65535,np.uint16)
        x,_=transform(counts,a,sa)
        self.assertEqual(x.dtype,np.float32)
        np.testing.assert_array_equal(x,np.log1p(counts.astype(np.float32)))

    def test_parameter_counts_and_initial_base(self):
        for t,expected in [(12,335244),(617,955369)]:
            model=CAPTI(t,{'rank':128,'base_scale':.5}).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),expected)
            b=torch.randn(4,t)
            with torch.no_grad():p=model(torch.rand(4,2048),torch.rand(4,3,152),b)
            torch.testing.assert_close(p,.5*b,atol=0,rtol=0)

    def test_masked_task_balance(self):
        z=torch.tensor([[0.,1.],[2.,-2.],[-1.,3.]],requires_grad=True)
        y=torch.tensor([[0.,1.],[1.,0.],[0.,0.]])
        mask=torch.tensor([[True,True],[True,False],[True,False]])
        cells=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
        expected=.5*(cells[:,0].mean()+cells[0,1])
        loss=masked_loss(z,y,mask);torch.testing.assert_close(loss,expected)
        loss.backward();self.assertEqual(z.grad[1:,1].abs().sum().item(),0.)

    def test_metrics_ties_missing_and_oneclass(self):
        y=np.array([[1,0,1],[0,np.nan,1],[1,1,np.nan],[0,1,1]],np.float32)
        p=np.array([[.3,.8,.2],[.3,.4,.4],[.8,.8,.1],[.1,.5,.2]],np.float32)
        r=metrics(y,p);self.assertEqual(r['eligible_task_ids'],[0,1])
        for row in r['per_task']:
            t=row['task'];m=np.isfinite(y[:,t])
            self.assertAlmostEqual(row['ap'],average_precision_score(y[m,t],p[m,t]),places=12)
            self.assertAlmostEqual(row['auroc'],roc_auc_score(y[m,t],p[m,t]),places=12)

    def test_alignment_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'split.npz'
            np.savez(p,train=[0,1],val=[1,2],y_train=[[1],[0]],y_val=[[0],[1]])
            with self.assertRaises(ValueError):load_split(p,3,['t'])
            p=Path(d)/'base.npz'
            np.savez(p,logits=np.zeros((2,1)),smiles=['C','N'],tasks=['t'])
            with self.assertRaises(ValueError):aligned_array(p,'logits',['N','C'],['t'])

    def test_synthetic_training_checkpoint(self):
        # Five toy optimization steps only; no real dataset or benchmark score.
        rng=np.random.default_rng(19);n=16;tasks=['a','b']
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);smiles=np.asarray([f'row_{i}' for i in range(n)])
            np.savez(d/'f.npz',count=rng.integers(0,4,(n,2048),dtype=np.uint16),
                     topology=rng.integers(0,4,(n,3,152)).astype(np.float32),smiles=smiles)
            y=np.column_stack([np.arange(n)%2,(np.arange(n)//2)%2]).astype(np.float32)
            np.savez(d/'s.npz',train=np.arange(12),val=np.arange(12,16),y_train=y[:12],y_val=y[12:])
            np.savez(d/'b.npz',logits=rng.normal(size=(n,2)).astype(np.float32),smiles=smiles,
                     tasks=tasks,train_ids=np.arange(12))
            config=json.loads((ROOT/'configs/default.json').read_text());config['epochs']=5
            r=fit(d/'f.npz',d/'s.npz',d/'b.npz',tasks,config,13,d/'out',device='cpu')
            self.assertEqual(r['epochs_run'],5)
            cp=torch.load(d/'out/head.pt',weights_only=False,map_location='cpu')
            CAPTI(2,cp['config']).load_state_dict(cp['state'],strict=True)
            self.assertTrue((d/'out/validation_predictions.npz').is_file())

if __name__=='__main__': unittest.main()
