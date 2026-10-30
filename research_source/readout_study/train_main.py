"""Only new F4 marginal heads and D600 OOF; all final heads once. XGBoost 2.1.4."""
import argparse, json, time, signal, resource, platform
import numpy as np
import xgboost as xgb
from common import *

def leaf_count(n): return 1 if 'leaf' in n else sum(leaf_count(c) for c in n['children'])
def fit(X,y,w,rows,evalrows,path,rounds):
    if path.exists(): raise FileExistsError(path)
    start=time.monotonic()
    dm=xgb.DMatrix(X[rows],label=y[rows],weight=w,nthread=1)
    m=xgb.train(PARAMS,dm,num_boost_round=rounds)
    path.parent.mkdir(parents=True,exist_ok=True);m.save_model(path)
    pred=m.inplace_predict(X[evalrows]).astype(np.float64)
    if rounds==600:
        np.save(path.with_suffix('.prefix150.npy'),m.inplace_predict(X[evalrows],iteration_range=(0,150)),allow_pickle=False)
    dumps=m.get_dump(dump_format='json')
    receipt={'model':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size,
             'trees':len(dumps),'leaves':sum(leaf_count(json.loads(t)) for t in dumps),
             'seconds':time.monotonic()-start,'train_rows':len(rows),'eval_rows':len(evalrows),
             'config':json.loads(m.save_config()),'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    savejson(path.with_suffix('.receipt.json'),receipt)
    print(json.dumps({k:receipt[k] for k in ['model','trees','leaves','seconds','rss_bytes']}),flush=True)
    return pred

def main():
    a=argparse.ArgumentParser();a.add_argument('--view',choices=['world','current'],required=True);a.add_argument('--stage',choices=['oof','final'],required=True);a=a.parse_args()
    assert xgb.__version__=='2.1.4'; signal.alarm(7200)
    z=dataset();X=z[a.view];ys=targets(z);N=len(X);start=time.time()
    out=ROOT/'results'/a.stage/a.view;out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions.npz').exists():raise FileExistsError(out/'predictions.npz')
    pred={}
    if a.stage=='oof':
        for key in ['p','ttc','lk','u600']:pred[key]=np.full(N,np.nan)
        for fold in range(3):
            layout=read(DATA/f'experiments/matched/v02/oof/fold{fold}/layout.npz')
            tr,ho,w=layout['train'],layout['hold'],layout['weights']
            assert np.array_equal(tr,tr[np.lexsort((z['candidate'][tr],z['scene_id'][tr]))])
            assert np.max(np.abs(w-weights(z,tr)))<1e-14
            assert not(set(z['drive'][tr])&set(z['drive'][ho]))
            for key,rounds in [('p',150),('ttc',150),('lk',150),('u600',600)]:
                y=ys['u' if key=='u600' else key]
                path=ROOT/f'models/oof/{a.view}/fold{fold}/{key}.ubj'
                pred[key][ho]=fit(X,y,w,tr,ho,path,rounds)
    else:
        rows=np.arange(N); rows=rows[np.lexsort((z['candidate'],z['scene_id']))];w=weights(z,rows)
        np.savez_compressed(out/'layout.npz',train=rows,weights=w)
        for key in ['q','p','ttc','lk','qp','qttc','qlk','u150','u600']:
            rounds=600 if key=='u600' else 150;y=ys['u' if key.startswith('u') else key]
            pred[key]=fit(X,y,w,rows,np.arange(N),ROOT/f'models/final/{a.view}/{key}.ubj',rounds)
    assert all(np.isfinite(v).all() for v in pred.values())
    np.savez_compressed(out/'predictions.npz',**pred)
    savejson(out/'completion.json',{'status':'PASS','view':a.view,'stage':a.stage,'start_unix':start,'end_unix':time.time(),'xgboost':xgb.__version__,'numpy':np.__version__,'platform':platform.platform(),'output_sha256':sha(out/'predictions.npz'),'scope':'New CPU fitting on fixed TRAIN240; final fitted values are NOT OOF evaluation.'})
if __name__=='__main__':main()
