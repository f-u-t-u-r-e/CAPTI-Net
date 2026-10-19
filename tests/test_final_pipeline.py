"""Synthetic software checks; these inputs are never manuscript observations."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from capti.model import CAPTI
from capti.variants import VARIANTS, variant_config
from capti.metrics import metrics
from capti.applicability import bin_indices, score_bins, nearest_training
from capti import predict, evaluate


class FinalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_full_model_equation(self):
        # Independent construction of the manuscript's algebra with trained-like weights.
        m=CAPTI(3,variant_config({'rank':4,'dropout':0.},'full')).eval()
        with torch.no_grad():
            for head in m.heads:head.weight.normal_()
            m.fragment_mean.uniform_(0,.02);m.bias.normal_()
        x=torch.rand(5,2048);c=torch.rand(5,3,152);base=torch.randn(5,3)
        f=torch.nn.functional.normalize(x,p=2,dim=1);center=f-m.fragment_mean
        first=f@m.pair
        products=.5*((center@m.pair)**2-(center**2)@(m.pair**2))
        a,b,d=[torch.nn.functional.silu(m.norms[i+2](m.scales[i](c[:,i]))) for i in range(3)]
        h_a=torch.cat([m.norms[0](first),m.norms[1](products)],1)
        h_t=torch.cat([a,b,d,*[m.norms[i+5](z) for i,z in enumerate([a*b,b*d,a*d])]],1)
        expected=.5*base+m.bias+m.heads[0](h_a)+m.heads[1](h_t)
        torch.testing.assert_close(m(x,c,base),expected,atol=2e-5,rtol=2e-5)

    def test_omitted_pathways_have_no_effect(self):
        x=torch.rand(3,2048);c=torch.rand(3,3,152);base=torch.randn(3,2)
        for name in VARIANTS:
            cfg=variant_config({'rank':4,'dropout':0.},name);m=CAPTI(2,cfg).eval()
            with torch.no_grad():
                for head in m.heads:head.weight.normal_()
            expected=m(x,c,base)
            if not cfg['i1']:torch.testing.assert_close(m(x*100,c,base),expected,atol=0,rtol=0)
            if not cfg['i2']:torch.testing.assert_close(m(x,c*100,base),expected,atol=0,rtol=0)
            if cfg['base_scale']==0:torch.testing.assert_close(m(x,c,base*100),expected,atol=0,rtol=0)

    def test_interaction_controls_retain_first_order(self):
        for name,head_index,first_width in [('no_abundance_products',0,4),('no_topology_products',1,12)]:
            m=CAPTI(2,variant_config({'rank':4,'dropout':0.},name)).eval();seen=[]
            hook=m.heads[head_index].register_forward_pre_hook(lambda mod,args:seen.append(args[0].detach()))
            m(torch.rand(3,2048),torch.rand(3,3,152),torch.rand(3,2));hook.remove()
            self.assertTrue(torch.count_nonzero(seen[0][:,:first_width])>0)
            self.assertEqual(torch.count_nonzero(seen[0][:,first_width:]),0)

    def test_metrics_tied_scores_against_sklearn(self):
        rng=np.random.default_rng(25)
        for _ in range(40):
            y=rng.integers(0,2,(35,5)).astype(float);y[rng.random(y.shape)<.2]=np.nan
            p=rng.integers(0,5,y.shape)/4
            for row in metrics(y,p)['per_task']:
                t=row['task'];mask=np.isfinite(y[:,t])
                self.assertAlmostEqual(row['ap'],average_precision_score(y[mask,t],p[mask,t]),places=12)
                self.assertAlmostEqual(row['auroc'],roc_auc_score(y[mask,t],p[mask,t]),places=12)

    def test_invalid_labels_and_probabilities(self):
        for y,p in [(np.array([[2.],[0.]]),np.zeros((2,1))),
                    (np.array([[1.],[0.]]),np.array([[np.nan],[.5]]))]:
            with self.assertRaises(ValueError):metrics(y,p)

    def test_similarity_boundaries(self):
        np.testing.assert_array_equal(bin_indices([0,.299,.3,.499,.5,.7,.701,1]),[0,0,1,1,2,2,3,3])
        with self.assertRaises(ValueError):bin_indices([1.1])

    def test_same_eligibility_and_empty_common_set(self):
        y=np.array([[0,0],[1,0],[0,0],[1,1],[0,0],[1,1],[0,0],[1,1]],float)
        p=np.arange(16).reshape(8,2)/16;s=np.repeat([.2,.4,.6,.8],2)
        rows,_=score_bins(y,{'a':p,'b':1-p},s)
        for r in rows:
            self.assertEqual(r['eligible_tasks'],1 if r['eligibility']=='common_across_bins' or r['bin']=='<0.3' else 2)
        rows,_=score_bins(y,{'a':p},np.full(8,.2))
        self.assertTrue(all(r['macro_ap'] is None for r in rows if r['eligibility']=='common_across_bins'))
        rows,_=score_bins(y.astype(np.int64),{'a':p},s)
        self.assertEqual(rows[0]['eligible_tasks'],1)

    def test_fdef_line_endings_and_modified_content(self):
        from capti.features import definition_sha256, check_environment, EXPECTED_FDEF, FDEF
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'BaseFeatures.fdef';source=FDEF.read_bytes().replace(b'\r\n',b'\n')
            path.write_bytes(source.replace(b'\n',b'\r\n'))
            self.assertEqual(definition_sha256(path),EXPECTED_FDEF)
            with patch('capti.features.FDEF',path):check_environment()
            path.write_bytes(source+b'\n# modified definition\n')
            with patch('capti.features.FDEF',path),self.assertRaises(RuntimeError):check_environment()

    def test_nearest_training_only(self):
        similarity,nearest=nearest_training(['CCO','c1ccccc1','OCC'],[0,1],[2])
        self.assertEqual(similarity.tolist(),[1.]);self.assertEqual(nearest.tolist(),[0])
        with self.assertRaises(ValueError):nearest_training(['C','N'],[0],[0])

    def test_predict_cli_outputs_final_p_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);inp=d/'m.json';out=d/'p.npz'
            inp.write_text(json.dumps({'smiles':['C','N'],'ids':[8,4]}))
            argv=['capti.predict','--molecules',str(inp),'--head','unused.pt','--output',str(out)]
            with patch('sys.argv',argv),patch.object(predict,'neural_predict',return_value=(np.array([[.2],[.8]]),['t'])):
                with contextlib.redirect_stdout(io.StringIO()):predict.main()
            with np.load(out,allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'p','ids','smiles','tasks'})
                np.testing.assert_array_equal(z['ids'],[8,4]);np.testing.assert_array_equal(z['p'],[[.2],[.8]])

    def test_checkpoint_mismatch_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'graph.pt').write_bytes(b'wrong checkpoint')
            torch.save({'config':{'base_scale':.5},'task_names':['a'],'graph_checkpoint_sha256':'0'*64},d/'head.pt')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):
                predict.neural_predict(['C'],d/'head.pt',d/'graph.pt')

    def test_chemical_only_predicts_without_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);cfg=variant_config({'rank':4},'topology_only');m=CAPTI(1,cfg)
            torch.save({'state':m.state_dict(),'config':cfg,'task_names':['a'],
                        'stats':{'pharmacophore_mean':np.zeros((3,152),np.float32),
                                 'pharmacophore_std':np.ones((3,152),np.float32)}},d/'head.pt')
            p,tasks=predict.neural_predict(['CCO','CCN'],d/'head.pt')
            np.testing.assert_array_equal(p,np.full((2,1),.5));self.assertEqual(tasks,['a'])

    def test_explicit_evaluation_task_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);np.savez(d/'y.npz',ids=[2,3],tasks=['a'],y=[[0],[1]])
            np.savez(d/'p.npz',ids=[2,3],tasks=['b'],p=[[.1],[.9]])
            argv=['capti.evaluate','--labels',str(d/'y.npz'),'--predictions',str(d/'p.npz'),'--output',str(d/'report.json')]
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'task order'):evaluate.main()
            np.savez(d/'p.npz',ids=[2,3],tasks=['a'],p=[[.1],[.9]])
            with patch('sys.argv',argv):evaluate.main()
            self.assertEqual(json.loads((d/'report.json').read_text())['macro_ap'],1.)


if __name__=='__main__':
    unittest.main()
