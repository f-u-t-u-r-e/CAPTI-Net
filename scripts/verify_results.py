"""Verify the released real-score snapshot without fitting or model selection."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd


def verify(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]/'results'
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
    folds=pd.read_csv(root/'partition_scores.csv');table=pd.read_csv(root/'table1.csv')
    assert len(folds)==110 and len(table)==22
    assert not folds.method.str.contains('rf|attentive',case=False).any()
    for row in table.itertuples():
        g=folds[(folds.dataset==row.dataset)&(folds.method==row.method+'_raw')].sort_values('seed')
        assert g.seed.tolist()==[13,23,37,53,71]
        for metric in ['macro_ap','macro_auroc']:
            assert np.isclose(g[metric].mean(),getattr(row,metric+'_mean'),atol=1e-12,rtol=0)
            assert np.isclose(g[metric].std(ddof=1),getattr(row,metric+'_sd'),atol=1e-12,rtol=0)
    for file,metric,delta in [('pathway_removal.csv','ablated_neural_ap','paired_delta_ap_full_minus_ablated_pp'),
                              ('single_pathways.csv','single_neural_ap','paired_delta_ap_full_minus_single_pp')]:
        data=pd.read_csv(root/file)
        for row in data.to_dict('records'):
            full=folds[(folds.dataset==row['dataset'])&(folds.seed==row['seed'])&(folds.method=='capti_raw')].iloc[0].macro_ap
            assert abs(full-row['full_neural_ap'])<1e-12
            assert abs((full-row[metric])*100-row[delta])<1e-10
    external=pd.read_csv(root/'external_herg.csv')
    assert len(external)==45 and set(external.panel)=={'neural'}
    expected={'external_I':(24,18),'external_II':(30,10),'external_III':(665,27)}
    for (cohort,method),g in external.groupby(['cohort','method']):
        assert sorted(g.seed)==[13,23,37,53,71]
        assert set(zip(g.n,g.positive))=={expected[cohort]}
    domain=pd.read_csv(root/'similarity_per_seed.csv')
    peers=domain[domain.method.isin(['CAPTI_neural','DMPNN_neural','MolCLR_neural'])]
    for (ds,mode,bin_),g in peers.groupby(['dataset','eligibility','bin']):
        if bin_=='all':continue
        assert (g.groupby('method')[['macro_ap','macro_auroc']].mean().idxmax()=='CAPTI_neural').all()
        for seed,part in g.groupby('seed'):
            assert (part[['molecules','eligible_tasks','macro_prevalence']].nunique()==1).all()
    return {'status':'passed','main_partition_rows':len(folds),'external_rows':len(external),
            'domain_rows':len(domain),'mean_and_sample_sd':'verified','paired_ablation_differences':'verified',
            'shared_domain_eligibility':'verified'}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
