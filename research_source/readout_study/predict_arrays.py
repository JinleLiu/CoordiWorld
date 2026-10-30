"""Portable CPU NumPy prediction from exported frozen tree arrays (no XGBoost).

Inputs are exactly the supplied W1689/C1260 FEATURE_SCHEMA_v02 matrices.
This returns mechanism scores, not the deployed PNC risk-qualified policy.
"""
from pathlib import Path
import json
import numpy as np

def project(y):
    y=np.asarray(y,dtype=np.float64);lo=np.zeros(len(y));hi=np.ones(len(y))
    for _ in range(60):
        q=(lo+hi)/2;neg=q-y[:,0]+np.minimum(q[:,None]-y[:,1:],0).sum(1)<0
        lo=np.where(neg,q,lo);hi=np.where(neg,hi,q)
    q=(lo+hi)/2;return np.c_[q,np.clip(y[:,1:],0,q[:,None])]

def tree_predict(path,X):
    X=np.asarray(X,dtype=np.float32)
    with np.load(path,allow_pickle=False) as t:
        if X.ndim!=2 or X.shape[1]!=int(t['features']):raise ValueError('feature dimension mismatch')
        if np.isinf(X).any():raise ValueError('infinite feature')
        n=len(X);out=np.full(n,t['base_score'].item(),dtype=np.float32)
        for k in range(len(t['left'])):
            node=np.zeros(n,dtype=np.int32)
            while True:
                active=t['left'][k,node]>=0
                if not active.any():break
                rr=np.flatnonzero(active);nn=node[rr];xx=X[rr,t['feature'][k,nn]]
                left=np.where(np.isnan(xx),t['default_left'][k,nn],xx<t['value'][k,nn])
                node[rr]=np.where(left,t['left'][k,nn],t['right'][k,nn])
            out+=t['value'][k,node]
    return out

def predict(model_dir,X,*,view,method,schema_sha256):
    model_dir=Path(model_dir);meta=json.loads((model_dir/'index.json').read_text())
    if schema_sha256!=meta['feature_schema_sha256']:raise ValueError('schema identity mismatch')
    X=np.asarray(X,dtype=np.float32)
    if X.ndim!=2 or X.shape[1]!=meta['dimensions'][view]:raise ValueError('feature dimension mismatch')
    def head(key):return tree_predict(model_dir/meta['models'][view][key]['arrays'],X).astype(np.float64)
    if method in ['D150','D600']:return head('u'+method[1:])
    hc=X[:,113].astype(np.float64);ec=X[:,301].astype(np.float64)
    if method=='F4':
        f=np.clip(np.stack([head(k) for k in ['q','p','ttc','lk']],axis=1),0,1)
        return f[:,0]*(5*f[:,1]+5*f[:,2]+2*f[:,3]+2*hc+2*ec)/16
    if method in ['Jraw','Jbox','Jproj']:qp='qp';mode=method[1:]
    elif method in ['Jlambda0_box','Jlambda0_proj','Jlambda1_box','Jlambda1_proj']:
        tag,mode=method.split('_');qp='qp_'+tag[1:]
    else:raise ValueError('unknown fixed readout')
    y=np.stack([head(k) for k in ['q',qp,'qttc','qlk']],axis=1)
    if mode=='box':y=np.clip(y,0,1)
    elif mode=='proj':y=project(y)
    return (2*(hc+ec)*y[:,0]+5*y[:,1]+5*y[:,2]+2*y[:,3])/16

def choose(scores,candidate_ids):
    scores=np.asarray(scores);ids=np.asarray(candidate_ids)
    if len(scores)==0 or len(scores)!=len(ids) or not np.isfinite(scores).all():raise ValueError('invalid candidates')
    order=np.argsort(ids,kind='stable');return int(ids[order[np.argmax(scores[order])]])
