"""V24 source-conditioned alternative-motion summaries, no future inputs.

These are alternative hypotheses, not calibrated event probabilities or new
world predictions. Frozen world/executor/candidate identity remain untouched.
"""
import argparse,time,datetime as dt
from revision_v5_eighteenth_common import *
from revision_v5_sixteenth_semantics import interpolate
from revision_v5_eighteenth_sat_fast import sat
NAMES=[]
for group in ['nonrail_vehicle_hold','vulnerable_hold','vulnerable_heading1mps']:
    NAMES += [group+'_'+q for q in ['supported','min_SAT_m','min_time_over4','first_contact_over4','contact_fraction','selected_current_speed','selected_world_displacement_m','selected_world_SAT_m']]
def features(a):
    ex=a['execution_full_local'];valid=a['candidate_valid'];cur=a['current'].astype(float);M,T=ex.shape[:2];N=len(cur);known=a['actor_valid']&a['current_known'];pred,pv=interpolate(a);eb=np.concatenate([ex[:,:,:3],np.broadcast_to(a['ego_box'],(M,T,2))],-1).copy();eb[:,:,:2]+=np.stack([np.cos(eb[:,:,2]),np.sin(eb[:,:,2])],-1)*float(a['rear_axle_offset'].ravel()[0]);f=[];tr=[]
    groups=[known&(a['types']==1)&(a['sources']!='physical'),known&np.isin(a['types'],[2,3]),known&np.isin(a['types'],[2,3])]
    for q,mask in enumerate(groups):
        ids=np.flatnonzero(mask);F=np.zeros((M,8));F[:,1]=50;F[:,2]=1;F[:,3]=1;F[:,7]=50
        if len(ids):
            aa=np.broadcast_to(cur[ids],(T,len(ids),7)).copy()
            if q==2:
                speed=np.maximum(np.linalg.norm(cur[ids,3:5],axis=-1),1.);vel=np.stack([np.cos(cur[ids,2]),np.sin(cur[ids,2])],-1)*speed[:,None];aa[:,:,:2]+=np.arange(T)[:,None,None]/10*vel[None]
            ab=np.concatenate([aa[:,:,:3],aa[:,:,5:7]],-1);g=sat(eb[:,:,None],ab[None]);flat=g.reshape(M,-1).argmin(1);tt,kk=np.unravel_index(flat,(T,len(ids)));jj=np.arange(M);contact=(g<=0).any(-1);sel=ids[kk];wf=np.concatenate([pred[tt,sel,:3],pred[tt,sel,5:7]],-1);wg=sat(eb[jj,tt],wf)
            F=np.stack([np.ones(M),np.clip(g[jj,tt,kk],-50,50),tt/40,np.where(contact.any(1),contact.argmax(1)/40,1),contact.mean(1),np.linalg.norm(cur[sel,3:5],axis=-1),np.linalg.norm(pred[tt,sel,:2]-aa[tt,kk,:2],axis=-1),np.clip(wg,-50,50)],-1);tr.append(dict(tokens=a['tokens'][sel].tolist(),min_step=tt.tolist(),supported_count=len(ids)))
        else:tr.append(dict(tokens=['']*M,min_step=[-1]*M,supported_count=0))
        f.append(F)
    out=np.concatenate(f,-1).astype(np.float32);assert np.isfinite(out).all();return np.where(valid[:,None],out,0),tr
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','dev'],default='train');args=p.parse_args();out=BASE/'features';out.mkdir(exist_ok=True);t=time.monotonic();ids=np.load(B16/'features'/('S64_'+args.phase+'.npz'))['scene_ids'];ff=[];trace=[]
    for i,s in enumerate(ids):
        a=dict(np.load(B16/'features'/('assets_'+args.phase)/str(s)/'assets.npz'));f,tr=features(a);ff.append(f);trace.append(dict(scene_id=str(s),groups=tr))
        if(i+1)%300==0:print('V24',args.phase,i+1,time.monotonic()-t,flush=True)
    np.savez_compressed(out/('V24_'+args.phase+'.npz'),scene_ids=ids,features=np.stack(ff),names=np.array(NAMES));write(out/('V24_'+args.phase+'_receipt.json'),dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha=sha(__file__),rows=len(ids),seconds=time.monotonic()-t,labels_used=False,source='current finite known actors + frozen world and41step execution; no nativefuture detections',heading_speed_floor_mps=1.,meaning='alternative motion stress summaries, not calibrated or guaranteed bounds',traces=trace))
if __name__=='__main__':main()
