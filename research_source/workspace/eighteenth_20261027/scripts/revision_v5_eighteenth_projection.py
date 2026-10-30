"""Exact equal-L2 projection onto the star order cone of four joint moments."""
import argparse,json,numpy as np
from revision_v5_eighteenth_common import *
def project(raw):
    y=np.stack([raw[k].astype(float)for k in ['q','qp','qttc','qlk']],-1);q=y[...,0].copy();ms=np.sort(np.maximum(y[...,1:],0),axis=-1)[...,::-1];total=q.copy();count=np.ones_like(q)
    for k in range(3):
        active=ms[...,k]>q;total+=np.where(active,ms[...,k],0);count+=active;q=total/count
    q=np.clip(q,0,1);m=np.clip(y[...,1:],0,q[...,None]);return np.concatenate([q[...,None],m],-1)
def validate():
    from scipy.optimize import minimize
    rng=np.random.default_rng(1818);y=rng.normal(.5,.8,(100,4));p=project({k:y[:,i]for i,k in enumerate(['q','qp','qttc','qlk'])});err=0
    for a,b in zip(y,p):
        rr=minimize(lambda x:.5*np.sum((x-a)**2),np.full(4,.5),jac=lambda x:x-a,bounds=[(0,1)]*4,constraints=[{'type':'ineq','fun':lambda x:x[0]-x[1:],'jac':lambda x:np.concatenate([np.ones((3,1)),-np.eye(3)],axis=1)}],method='SLSQP',options={'ftol':1e-12,'maxiter':200});assert rr.success;err=max(err,float(abs(rr.x-b).max()))
    assert err<1e-6;return dict(max_abs_vs_SLSQP=err,feasible=True,exact_same_if_input_feasible=True,decision_guarantee=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--math-only',action='store_true');a=ap.parse_args()
    if a.math_only:print(json.dumps(validate()));return
    out=BASE/'exploration/IP';out.mkdir(parents=True,exist_ok=True);z,_=load();result={};write(out/'math_check.json',validate())
    for phase in ['oof','robust']:
        old=parent(phase);nr=np.load(BASE/'runs/NR'/phase/'oof.npz');result[phase]={}
        for method in ['NR','QV']:
            pp=old if method=='NR'else dict(np.load(BASE/'runs/QV'/phase/'oof.npz'));raw={h:pp['raw_'+h]for h in ['q','qp','qttc','qlk']};p=project(raw);score=(5*p[...,1]+5*p[...,2]+2*p[...,3]+2*z['features'][...,197]*p[...,0])/14;ch,el=select(score,z['valid'],z['choice_Geometry_HC'],nr['risk'],old['raw_dac_event']);reference=nr['choice']if method=='NR'else pp['choice'];m=metrics(ch,z);np.savez_compressed(out/(method+'_'+phase+'.npz'),scene_ids=z['scene_ids'],choice=ch,eligible=el,scores=score,projected=p,**{'raw_'+k:v for k,v in raw.items()});result[phase][method]=dict(metrics=m,gate=gate(m),vsControl=comparison(ch,reference,z),vsGeometry=comparison(ch,z['choice_Geometry_HC'],z),projection_changed_rows=int((z['valid']&(p[...,0]!=np.clip(raw['q'],0,1))).sum()));print('IP',method,phase,m,flush=True)
    write(out/'result.json',result)
if __name__=='__main__':main()
