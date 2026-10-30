"""R16 fixed-data readout experiment. No simulator, future generation or selector gates."""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'startup/CoordiWorld_再来一次_R16_启动包_v02_20261030/portable_TRAIN240_readout'
PARAMS = dict(objective='reg:squarederror', tree_method='hist', max_leaves=15,
              max_depth=0, grow_policy='lossguide', eta=.05, reg_lambda=10,
              min_child_weight=5, seed=17, subsample=1, colsample_bytree=1,
              nthread=1, disable_default_eval_metric=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def savejson(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def dataset(): return read(DATA/'data/train240/matched_dataset_v02.npz')
def targets(z):
    q = np.prod(z['components'][:,:4],axis=1)
    p,t,l = z['components'][:,4:7].T
    return dict(q=q,p=p,ttc=t,lk=l,qp=q*p,qttc=q*t,qlk=q*l,u=z['U'])
def weights(z, rows):
    scenes=z['scene_id'][rows]; drives=z['drive'][rows]
    ns={s:int((scenes==s).sum()) for s in np.unique(scenes)}
    nd={d:len(np.unique(scenes[drives==d])) for d in np.unique(drives)}
    w=np.array([1/(ns[s]*nd[d]) for s,d in zip(scenes,drives)],dtype=np.float64)
    return w/w.mean()
def project(y):
    y=np.asarray(y,dtype=np.float64); q=y[:,0]; kids=y[:,1:]
    lo=np.zeros(len(q));hi=np.ones(len(q))
    for _ in range(60):
        mid=(lo+hi)/2; neg=mid-q+np.minimum(mid[:,None]-kids,0).sum(1)<0
        lo=np.where(neg,mid,lo);hi=np.where(neg,hi,mid)
    q=(lo+hi)/2
    return np.c_[q,np.clip(kids,0,q[:,None])]
def coefficients(z):
    return np.c_[2*(z['components'][:,7]+z['EC']), np.full(len(z['U']),5),
                 np.full(len(z['U']),5),np.full(len(z['U']),2)]/16
def readouts(z,j,f,d150,d600):
    c=coefficients(z); fb=np.clip(f,0,1)
    return {'D150':d150,'D600':d600,'F4':fb[:,0]*(5*fb[:,1]+5*fb[:,2]+2*fb[:,3]+2*z['components'][:,7]+2*z['EC'])/16,
            'Jraw':(j*c).sum(1),'Jbox':(np.clip(j,0,1)*c).sum(1),'Jproj':(project(j)*c).sum(1)}
