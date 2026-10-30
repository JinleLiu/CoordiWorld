"""B18 frozen data and evaluation contracts."""
from pathlib import Path
import numpy as np,json
from revision_v5_sixteenth_common import *
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));BASE=ROOT/'eighteenth_20261027';B17=ROOT/'seventeenth_20261027';B16=ROOT/'sixteenth_20261027'
def load():
    z=dict(np.load(B16/'train_inputs.npz'));p=np.load(B17/'progress/P16_train.npz');s=np.load(B16/'features/S64_train.npz');np.testing.assert_array_equal(z['scene_ids'],p['scene_ids']);x=np.concatenate([pair_features(z['features'],z['valid']),s['features'],p['features']],-1).astype(np.float32);assert x.shape==(1755,12,674);return z,x
def parent(phase):return dict(np.load(B17/'runs/E0'/phase/'oof.npz'))
def select(scores,valid,geo,nc=None,dac=None,safe_score=False):
    eligible=valid.copy();ii=np.arange(len(valid))
    if nc is not None:eligible&=nc>=nc[ii,geo,None]if safe_score else nc<=nc[ii,geo,None]
    if dac is not None:eligible&=dac<=dac[ii,geo,None]
    assert eligible[ii,geo].all();return choose(scores,eligible),eligible
def diagnose(pred,score,valid,geo,z,choice,eligible):
    d=z['drives'];ids=z['scene_ids'];c=z['components'];ev=c[:,:,0]<1;ii=np.arange(len(valid));par=choose(score,valid);delta=z['utility'][ii,par].astype(float)-z['utility'][ii,geo].astype(float);rej=~eligible[ii,par];good=rej&(delta>0);all_safe=(~ev|~valid).all(1);all_event=(ev|~valid).all(1);mix=~all_safe&~all_event;pair=[];wrong=[];ties=[];pairdr=[]
    for i in np.flatnonzero(mix):
        safe=np.flatnonzero(valid[i]&~ev[i]);event=np.flatnonzero(valid[i]&ev[i]);gap=pred[i,event,None]-pred[i,safe][None,:];pair.append(float((gap>0).mean()+.5*(gap==0).mean()));wrong.append(int((gap<0).sum()));ties.append(int((gap==0).sum()));pairdr.append(d[i])
    return dict(mixed_scenes=int(mix.sum()),pair_accuracy_scene_drive_macro=macro(np.array(pair),np.array(pairdr)),wrong_pairs=sum(wrong),tie_pairs=sum(ties),beneficial_parent_rejected=int(good.sum()),beneficial_parent_rejected_points=100*macro(np.where(good,delta,0),d),beneficial_parent_rejected_ids=ids[good].tolist(),all_safe_scenes=int(all_safe.sum()),all_safe_parent_rejected=int((all_safe&rej).sum()),all_safe_delta_U_vs_parent=100*macro(np.where(all_safe,z['utility'][ii,choice]-z['utility'][ii,par],0),d),all_event_scenes=int(all_event.sum()),eligible_singleton_scenes=int((eligible.sum(1)==1).sum()))
