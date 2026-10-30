"""One qp-only lambda 0/1 comparison with shared DMatrix and explicit intercept."""
import argparse, time, json, signal, resource
import xgboost as xgb
from common import *
from train_main import leaf_count

def derivatives(pred, y, w, group, lam):
    e=np.asarray(pred,dtype=np.float64)-y
    counts=np.bincount(group);means=np.bincount(group,weights=e)/counts
    c=1/(1+lam)
    grad=c*w*(e+lam*(e-means[group]))
    # Diagonal PSD upper bound of the full within-scene coupled Hessian.
    upper=c*w*(1+lam)
    return grad,upper
def objective_value(pred,y,w,group,lam):
    e=pred-y;means=np.bincount(group,weights=e)/np.bincount(group)
    return .5/(1+lam)*np.sum(w*(e*e+lam*(e-means[group])**2))
def check():
    p=np.array([-.1,.6,.8,.2,.3,.7]);y=np.array([0,.2,.9,.4,.7,.1]);g=np.array([0,0,0,1,1,2]);w=np.array([.4,.4,.4,1.2,1.2,.8]);out={}
    for lam in [0,1]:
        grad,upper=derivatives(p,y,w,g,lam);numeric=[];eps=1e-6
        for i in range(len(p)):
            d=np.zeros(len(p));d[i]=eps;numeric.append((objective_value(p+d,y,w,g,lam)-objective_value(p-d,y,w,g,lam))/(2*eps))
        err=float(np.max(abs(grad-numeric)));assert err<1e-9
        mineig=1e9;pairerr=0
        for s in np.unique(g):
            ix=np.flatnonzero(g==s);n=len(ix);ws=w[ix[0]];e=p[ix]-y[ix]
            h=ws/(1+lam)*((1+lam)*np.eye(n)-lam*np.ones((n,n))/n)
            mineig=min(mineig,float(np.linalg.eigvalsh(np.diag(upper[ix])-h).min()))
            pairerr=max(pairerr,abs(np.sum((e-e.mean())**2)-sum((e[i]-e[j])**2 for i in range(n) for j in range(i+1,n))/n))
        assert mineig>=-1e-12 and pairerr<1e-12
        out[str(lam)]={'gradient_max_error':err,'upper_minus_hessian_min_eigenvalue':mineig,'pair_identity_error':float(pairerr)}
    savejson(ROOT/'results/custom-math-check.json',out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--view',choices=['world','current']);ap.add_argument('--stage',choices=['oof','final'],default='oof');ap.add_argument('--check',action='store_true');a=ap.parse_args()
    if a.check:check();return
    signal.alarm(7200);assert xgb.__version__=='2.1.4'
    z=dataset();X=z[a.view];y=targets(z)['qp'];N=len(y);start=time.time()
    out=ROOT/f'results/custom{("-final" if a.stage=="final" else "")}/{a.view}';out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions.npz').exists():raise FileExistsError(out/'predictions.npz')
    pred={f'lambda{lam}':np.full(N,np.nan) for lam in [0,1]}
    for fold in (range(3) if a.stage=='oof' else ['final']):
        if fold=='final':
            tr=np.arange(N);tr=tr[np.lexsort((z['candidate'],z['scene_id']))];ho=np.arange(N);w=weights(z,tr)
        else:
            layout=read(DATA/f'experiments/matched/v02/oof/fold{fold}/layout.npz');tr,ho,w=layout['train'],layout['hold'],layout['weights']
        _,groups=np.unique(z['scene_id'][tr],return_inverse=True)
        assert all(np.ptp(w[groups==s])==0 for s in np.unique(groups))
        dm=xgb.DMatrix(X[tr],label=y[tr],weight=w,nthread=1)
        yy=dm.get_label().astype(np.float64);ww=dm.get_weight().astype(np.float64)
        base=float(np.average(yy,weights=ww));params=dict(PARAMS,base_score=base,boost_from_average=0)
        for lam in [0,1]:
            path=ROOT/f'models/custom/{a.view}/{fold}/qp_lambda{lam}.ubj'
            if path.exists():raise FileExistsError(path)
            def obj(p,d):
                grad,hess=derivatives(p,yy,ww,groups,lam)
                return grad.astype(np.float32),hess.astype(np.float32)
            t=time.monotonic();m=xgb.train(params,dm,num_boost_round=150,obj=obj)
            path.parent.mkdir(parents=True,exist_ok=True);m.save_model(path);pred[f'lambda{lam}'][ho]=m.inplace_predict(X[ho])
            receipt={'lambda':lam,'base_score':base,'curvature_scale':1/(1+lam),'trees':150,'leaves':sum(leaf_count(json.loads(t)) for t in m.get_dump(dump_format='json')),'bytes':path.stat().st_size,'sha256':sha(path),'seconds':time.monotonic()-t,'config':json.loads(m.save_config()),'single_weight_application':True,'same_DMatrix_for_lambda_pair':True,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            savejson(path.with_suffix('.receipt.json'),receipt);print(json.dumps({'model':str(path.relative_to(ROOT)),'seconds':receipt['seconds'],'leaves':receipt['leaves']}),flush=True)
    assert all(np.isfinite(p).all() for p in pred.values())
    np.savez_compressed(out/'predictions.npz',**pred)
    savejson(out/'completion.json',{'status':'PASS','start_unix':start,'end_unix':time.time(),'prediction_sha256':sha(out/'predictions.npz'),'other_heads':'Reuse exact J4 q,qttc,qlk arrays; only qp differs','stage':a.stage})
if __name__=='__main__':main()
