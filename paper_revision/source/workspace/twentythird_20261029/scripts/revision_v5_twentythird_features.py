"""Add one trajectory with the frozen reference world and current feature view.
The existing12 proposal trajectories and reference remain unchanged. A spare
temporary slot is used only to call the pinned12-wide world feature kernel.
"""
from revision_v5_twentythird_common import *
import numpy as np,copy,time,torch
from revision_v5_seventeenth_live import Pipeline,b
from coordiworld.data.openscene import make_inputs,frame_agents,ego_pose
from revision_v5_sixteenth_semantics import semantic_features,current_areas
from revision_v5_twentysecond_kinematics import fix_s64,recover_legacy
from revision_v5_twentysecond_contact_features import contact_features
from revision_v5_eighteenth_source_features import features as vf
from revision_v5_eighteenth_road import features as rf
from revision_v5_seventeenth_math import physical_progress_proxy
def p_context(distance,pdm_distance,remaining,known,legacy_q,legacy_known,valid,ref,geo):
    M=len(valid);p0,k=physical_progress_proxy(distance,pdm_distance,np.full(M,known));p0=np.where(k,p0,np.where(legacy_known,legacy_q,0));d=np.where(k,distance,0);b=np.full(M,pdm_distance if known else 0.);den=np.maximum(d,b)
    vals=[d/100,b/100,den/100,(d-b)/100,p0,np.where(k,den<=5,0),(d-d[ref])/100,p0-p0[ref],(d-d[geo])/100,p0-p0[geo],k,legacy_known,legacy_q,~k&legacy_known,~k&~legacy_known,np.where(k,remaining,0)/100]
    f=np.stack(vals,-1).astype(np.float32);f[~valid]=0;p0=np.where(valid,p0,0);assert f.shape==(M,16)and np.isfinite(f).all();return f,p0

from coordiworld.revision_v5.component_known_b13 import geometry_known_score

def add_candidate(pipe,history,source_sha,a,assets,current,candidate):
 t=time.perf_counter();ref=int(a['valid'].argmax());slot=11 if ref!=11 else 10;cfg=copy.deepcopy(pipe.config);objects=frame_agents(history[-1],np.asarray(ego_pose(history[-1])));cfg['data']['max_agents']=max(1,sum(np.linalg.norm(s[:2])<=cfg['data']['roi_m']for s,_ in objects.values()));scene,meta=make_inputs(history,cfg,pipe.provider(history,cfg['data']['max_map_tokens']),pipe.proposer(history));assert meta['truncated_agents']==0
 cand=a['candidates'][:12].copy();vv=a['valid'][:12].copy();cand[slot]=candidate;vv[slot]=True;scene.candidates=torch.from_numpy(cand[None].astype(np.float32));scene.candidate_valid=torch.from_numpy(vv[None]);start=time.perf_counter();new=pipe.inputs(history,source_sha,scene,meta);forward_seconds=time.perf_counter()-start
 assert new['assets']['world'].shape==assets['world'].shape
 err=float(np.max(np.abs(new['assets']['world']-assets['world']),initial=0));assert err<1e-5,err
 # Derived features use the exact saved common world; only execution is new.
 aa={k:v.copy()for k,v in assets.items()};aa['execution']=np.concatenate([assets['execution'],new['assets']['execution'][slot:slot+1]]);aa['execution_full_local']=np.concatenate([assets['execution_full_local'],new['assets']['execution_full_local'][slot:slot+1]]);valid=np.r_[a['valid'],True];aa['candidate_valid']=valid
 area,_=current_areas(current,aa['execution_full_local'],ROOT/'data/maps');ss,_,trace=semantic_features(aa,area);corrected,_=fix_s64(aa,current,ss,trace);cf,_=contact_features(aa,current,area)
 f=np.concatenate([a['features198'],new['features198'][slot:slot+1]]);g=int(np.where(valid,geometry_known_score(f[None],f[None,:,188:],valid[None])[0],-np.inf).argmax());oldp=a['x674'][:,658:];dist=np.r_[oldp[:,0]*100,new['P16'][slot,0]*100];remaining=np.r_[oldp[:,15]*100,new['P16'][slot,15]*100];pp,_=p_context(dist,float(oldp[ref,1]*100),remaining,bool(oldp[ref,10]),f[:,196],f[:,195].astype(bool)&valid,valid,ref,g)
 pp[:-1]=a['x674'][:,658:]
 if g!=int(a['geo']):pp[:,8]=pp[:,0]-pp[g,0];pp[:,9]=pp[:,4]-pp[g,4]
 pair=np.concatenate([f,np.broadcast_to(f[ref],f.shape),f-f[ref]],-1);s64=np.concatenate([a['x674'][:,594:658],corrected[-1:]]);legacy=np.concatenate([a['old_x674'][:,594:658],ss[-1:]])
 out=dict(candidates=np.concatenate([a['candidates'],candidate[None]]),features198=f,x674=np.concatenate([pair,s64,pp],-1).astype(np.float32),old_x674=np.concatenate([pair,legacy,pp],-1).astype(np.float32),V24=np.concatenate([a['V24'],vf(aa)[0][-1:]]),R8=np.concatenate([a['R8'],rf(area,valid)[-1:]]),C32=np.concatenate([a['C32'],cf[-1:]]),valid=valid,geo=np.array(g))
 native=recover_legacy(aa['execution_full_local'],current['rear_axle_pose']);rec=dict(seconds=time.perf_counter()-t,world_pipeline_seconds=forward_seconds,world_maxerr=err,reference=ref,new_index=len(valid)-1,old_candidates_preserved=True,world='exact frozen old bytes',P16='candidate/PDM pair denominator unchanged; pool Geometry context updated')
 return out,aa,native,rec

def replace_from_appended(a,assets,index):
 out={}
 for k,v in a.items():
  if k=='geo':continue
  vv=np.asarray(v).copy();vv[index]=vv[-1];out[k]=vv[:-1]
 aa={k:v.copy()for k,v in assets.items()}
 for k in ['execution','execution_full_local','candidate_valid']:aa[k][index]=aa[k][-1];aa[k]=aa[k][:-1]
 f=out['features198'];v=out['valid'];g=int(np.where(v,geometry_known_score(f[None],f[None,:,188:],v[None])[0],-np.inf).argmax());out['geo']=np.array(g);ref=int(v.argmax());p=out['x674'][:,658:];p16,_=p_context(p[:,0]*100,float(p[ref,1]*100),p[:,15]*100,bool(p[ref,10]),f[:,196],f[:,195].astype(bool)&v,v,ref,g)
 for k in ['x674','old_x674']:out[k][:,658:]=p16
 return out,aa
