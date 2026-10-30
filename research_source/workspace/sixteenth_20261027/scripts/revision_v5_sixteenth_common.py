"""B16 fixed drive-macro evaluation; no candidate/metric modifications."""
from pathlib import Path
import hashlib,json,numpy as np
COMP=['no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance','ego_progress','time_to_collision_within_bound','lane_keeping','history_comfort']
EVENT={'NC':0,'DAC':1,'DDC':2,'TLC':3,'TTC':5,'LK':6,'HC':7}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def macro(x,d):return float(np.mean([np.asarray(x,dtype=float)[d==s].mean()for s in sorted(set(d))]))
def bootstrap(x,d):
    v=np.array([np.asarray(x,dtype=float)[d==s].mean()for s in sorted(set(d))]);r=np.random.default_rng(102611);return np.quantile(v[r.integers(len(v),size=(10000,len(v)))].mean(1),[.025,.975]).tolist()
def choose(s,v):return np.where(v,s,-np.inf).argmax(1)
def pair_features(x,v):
    ref=x[np.arange(len(x)),v.argmax(1),None];diff=x-ref;assert (diff[np.arange(len(x)),v.argmax(1)]==0).all();return np.concatenate([x,np.broadcast_to(ref,x.shape),diff],-1)
def metrics(choice,z,indices=None):
    ix=np.arange(len(z['valid']))if indices is None else np.asarray(indices);d=z['drives'][ix];v=z['valid'][ix];u=z['utility'][ix].astype(float);c=z['components'][ix];j=np.arange(len(ix));cc=c[j,choice].astype(float);uu=u[j,choice];oracle=np.where(v,u,-np.inf).max(1)
    return dict(scenes=len(ix),drives=len(set(d)),U=100*macro(uu,d),P=macro(cc[:,4],d),oracle_regret=100*macro(oracle-uu,d),**{e:int((cc[:,k]<1).sum())for e,k in EVENT.items()})
def comparison(choice,reference,z):
    i=np.arange(len(choice));d=z['drives'];ids=z['scene_ids'];u=z['utility'].astype(float);c=z['components'];du=100*(u[i,choice]-u[i,reference]);dp=c[i,choice,4].astype(float)-c[i,reference,4].astype(float)
    drives={k:float(du[d==k].mean())for k in sorted(set(d))};counts={k:int((d==k).sum())for k in drives};contrib=du/np.array([counts[k]for k in d])/len(drives)
    out=dict(delta_U=macro(du,d),CI95=bootstrap(du,d),delta_P=macro(dp,d),P_CI95=bootstrap(dp,d),changes=int((choice!=reference).sum()),by_drive=drives,leave_one_drive={k:float(np.mean([v for j,v in drives.items()if j!=k]))for k in drives},scene_contribution_points=contrib.tolist())
    for e,k in EVENT.items():
        aa=c[i,choice,k]<1;bb=c[i,reference,k]<1;out[e]=dict(method_total=int(aa.sum()),reference_total=int(bb.sum()),rescued_ids=ids[bb&~aa].tolist(),new_ids=ids[aa&~bb].tolist())
    return out
def gate(m):return bool(m['U']>=93.19763899051931 and m['NC']<=12 and m['DAC']<=8 and m['P']>=.8838486224992207)
def rank_key(m,complexity):
    good=m['NC']<=12 and m['DAC']<=8 and m['P']>=.8838486224992207
    return (0,-m['U'],complexity)if good else(1,max(m['NC']-12,0)+max(m['DAC']-8,0),max(.8838486224992207-m['P'],0),-m['U'],complexity)
