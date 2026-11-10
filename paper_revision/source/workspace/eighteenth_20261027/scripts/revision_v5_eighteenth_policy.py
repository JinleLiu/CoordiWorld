"""Portable frozen B18 decision interface; never consumes labels or outcomes."""
import numpy as np,json
from pathlib import Path
import xgboost as xgb
from revision_v5_eighteenth_common import ROOT,BASE,B17,sha,select
from revision_v5_seventeenth_train import compose
from revision_v5_eighteenth_projection import project
HEADS=['q','qp','qttc','qlk']
class Policy:
    def __init__(self,policy,phase='final',fold=None):
        self.policy=policy;self.models={};self.bindings={};self.sub='fit'if phase=='final'else'fold'+str(fold);self.phase=phase
        old=json.load(open(B17/'runs/E0'/phase/self.sub/'bindings.json'))
        for h in HEADS+['dac_event']:
            assert not old[h]['old_input'];self.add(h,Path(old[h]['path']),None)
        self.add('nc_rank',BASE/'runs'/policy.get('nc','NR')/phase/self.sub/'model.ubj',policy.get('nc_features'))
        value=policy.get('value','J4P')
        if value=='MT':self.add('vector',BASE/'runs/MT'/phase/self.sub/'model.ubj',None)
        elif value!='J4P':
            r=next(r for r in json.load(open(BASE/'EXPLORATION.json'))['recipes']if r['id']==value)
            binding=json.load(open(BASE/'runs'/value/phase/self.sub/'bindings.json'))
            for h,b in binding.items():self.add(h,Path(b['path']),r.get('features'))
        if policy.get('dac')=='DR':self.add('dac_event',BASE/'runs/DR'/phase/self.sub/'model.ubj','R8')
        self.calibrators=json.load(open(BASE/'runs/IC'/phase/self.sub/'calibrators.json'))if policy.get('isotonic')else None
    def add(self,h,path,extra):
        m=xgb.Booster();m.load_model(path);self.models[h]=(m,extra);self.bindings[h]=dict(path=str(path),sha256=sha(path),extra_input=extra)
    def predict(self,x,features198,valid,geo,extras=None):
        extras=extras or{};raw={};x=np.asarray(x,np.float32);assert x.shape[-1]==674
        for h,(m,extra)in self.models.items():
            if 'vector'in self.models and h in HEADS:continue
            xx=x if extra is None else np.concatenate([x,extras[extra]],-1).astype(np.float32);v=m.inplace_predict(xx.reshape(-1,xx.shape[-1]));raw[h]=v.reshape(*valid.shape,4)if h=='vector'else v.reshape(valid.shape)
        if 'vector'in raw:
            for j,h in enumerate(HEADS):raw[h]=raw['vector'][...,j]
        score,proj,before=compose(raw,0,features198[...,197],'PD')
        if self.policy.get('projection')=='IP':
            p=project(raw);proj={h:p[...,j]for j,h in enumerate(HEADS)};score=(5*p[...,1]+5*p[...,2]+2*p[...,3]+2*features198[...,197]*p[...,0])/14
        nr=-raw['nc_rank'];dr=raw['dac_event'];nc=nr;dac=dr
        if self.calibrators:
            nc=np.interp(nr,self.calibrators['NC']['x'],self.calibrators['NC']['y']);dac=np.interp(dr,self.calibrators['DAC']['x'],self.calibrators['DAC']['y'])
        choice,eligible=select(score,valid,geo,nc,dac)
        return dict(scores=score,choice=choice,eligible=eligible,risk=nc,dac=dac,uncalibrated_nc_rank_risk=nr,uncalibrated_dac=dr,qp_before_clip=before,**{'raw_'+h:raw[h]for h in HEADS},**{'projected_'+h:v for h,v in proj.items()})
