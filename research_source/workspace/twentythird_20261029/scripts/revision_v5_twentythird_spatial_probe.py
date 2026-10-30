"""TRAIN branch: distance-domain current-state join plus own acceleration replay.
Adapts A R12 spatial_join/quintic_offset boundary algebra, not A's selector.
"""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_factory import extend,quintic,wrap,ec_matrix
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
from revision_v5_seventeenth_live import b
import numpy as np,time,argparse,copy

def spatial_dynamic(current,previous,offset):
 ref,meta=extend(previous,offset);c=np.asarray(current);rot=np.array([[np.cos(c[2]),-np.sin(c[2])],[np.sin(c[2]),np.cos(c[2])]]);xy=(ref[:,:2]-c[:2])@rot;head=np.unwrap(ref[:,2]-c[2]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];valid=np.r_[True,np.diff(arc)>1e-5];t=np.arange(41)*.1;v0=max(0.,float(c[3]));k0=c[9]/max(v0,.5);theta=head[0];tangent=np.array([np.cos(theta),np.sin(theta)]);kref=ref[0,9]/max(abs(ref[0,3]),.5);length=max(8.,2*v0,2*abs(float(xy[0]@np.array([-np.sin(theta),np.cos(theta)]))));corrected=xy+quintic(-xy[0],np.array([1.,0.])-tangent,np.array([0.,k0])-kref*np.array([-np.sin(theta),np.cos(theta)]),length,arc);corrected[0]=0
 # Reparameterize the corrected curve by actual geometric distance so the
 # acceleration profile is not silently changed by boundary deformation.
 ds=np.linalg.norm(np.diff(corrected,axis=0),axis=1);s=np.r_[0,np.cumsum(ds)];keep=np.r_[True,ds>1e-6];curve=corrected[keep];s=s[keep];yaw=np.unwrap(np.arctan2(np.gradient(corrected,axis=0)[:,1],np.gradient(corrected,axis=0)[:,0]));yaw[0]=0.;hy=yaw[keep];v=v0;distance=0;rows=[];tail=0
 for it in range(1,401):
  tm=it*.01;u=min(tm/2.,1.);fade=1-3*u*u+2*u*u*u;a=float(np.interp(tm,t,ref[:,5]))+(c[5]-ref[0,5])*fade;vn=max(0.,v+.01*a);distance+=(v+vn)*.005;v=vn
  if it%50==0:
   if len(s)>1 and distance<=s[-1]:pos=np.array([np.interp(distance,s,curve[:,j])for j in [0,1]]);th=np.interp(distance,s,hy)
   else:th=hy[-1];tail=max(tail,distance-s[-1]);pos=curve[-1]+(distance-s[-1])*np.array([np.cos(th),np.sin(th)])
   rows.append([*pos,wrap(th)])
 return np.array(rows,np.float32),dict(**meta,spatial_join_length_m=length,additional_path_tail_m=float(tail),curve_reparameterized=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='oof');args=ap.parse_args();phase=args.phase;src=BASE/'probe80'/phase;out=BASE/'spatial80'/phase;out.mkdir(parents=True,exist_ok=True);reg=json.load(open(src/'result.json'));last=None;rows=[];start=time.monotonic();record('spatial_dynamic_'+phase,'Dynamic compatible proposals contain3DAC failures; time-indexed curvature may mismatch current travel distance','A-inspired distancequintic currentpose/tangent/curvature join, arc reparameterization, same longitudinal dynamic2 profile','same80TRAIN ownPNC history; mainDYN2 remainsfrozen')
 for r in reg['rows']:
  sid=r['scene_id'];d=out/sid;d.mkdir(exist_ok=True);a=np.load(src/sid/'inputs.npz');cur=a['old_native'][int(a['current_valid'].argmax()),0];cand,detail=spatial_dynamic(cur,a['previous_native'],r['offset']);s=r['source']
  if last!=s['log_name']:frames=b.load_log(ROOT/'data/openscene/trainval'/(s['log_name']+'.pkl'));last=s['log_name']
  h=frames[s['index']-3:s['index']+1];_,full,_=predict_vector_controller_execution(h,cand[None],np.ones(1,bool));ex=full[0][None];ec=ec_matrix(ex,a['previous_native'],r['offset']);np.savez_compressed(d/'inputs.npz',new_candidates=cand[None],new_native=ex,new_EC=ec);rr=copy.deepcopy(r);rr.update(variants=['spatial_dynamic2'],new_EC=ec.tolist(),details={'spatial_dynamic2':detail});write(d/'record.json',rr);rows.append(rr)
 write(out/'result.json',dict(complete=True,created_utc=now(),rows=rows,seconds=time.monotonic()-start,counts=dict(on_trigger=sum(r['trigger']and r['new_EC'][0]>0 for r in rows))));print('B23_SPATIAL',json.load(open(out/'result.json'))['counts'],flush=True)
if __name__=='__main__':main()
