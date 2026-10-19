"""Observed-label macro metrics; require both classes per task."""
import numpy as np

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if y.ndim!=2 or y.shape!=p.shape or np.isinf(y).any():
        raise ValueError('Labels and probabilities must be matching matrices; missing labels use NaN.')
    if not np.isin(y[np.isfinite(y)],[0,1]).all():
        raise ValueError('Observed labels must be 0 or 1.')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Predictions must be finite probabilities in [0,1].')
    ap=[];auc=[];eligible=[];per=[]
    for t in range(y.shape[1]):
        mask=np.isfinite(y[:,t]);yt=y[mask,t];pt=p[mask,t]
        if len(yt) and np.unique(yt).size==2:
            # Exact threshold grouping, including prediction ties, matches
            # sklearn AP/AUROC while avoiding hundreds of estimator wrappers.
            order=np.argsort(-pt,kind='stable');sy=yt[order];sp=pt[order]
            end=np.r_[np.flatnonzero(sp[:-1]!=sp[1:]),len(sp)-1]
            tp=np.cumsum(sy,dtype=np.float64)[end];n=end+1
            positive=float(sy.sum());negative=len(sy)-positive
            increments=np.diff(np.r_[0.,tp])
            a=float(np.sum(increments*(tp/n))/positive)
            tpr=np.r_[0.,tp/positive];fpr=np.r_[0.,(n-tp)/negative]
            u=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])*.5))
            ap.append(a);auc.append(u);eligible.append(t)
            per.append({'task':t,'ap':a,'auroc':u,'n':len(yt),'positives':int(yt.sum())})
    return {'macro_ap':float(np.mean(ap)) if ap else None,'macro_auroc':float(np.mean(auc)) if auc else None,
            'eligible_tasks':len(ap),'eligible_task_ids':eligible,'per_task':per}

