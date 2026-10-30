"""Matched readout inputs v02. C never reads predicted other-actor states.

C is conditional on the supplied common bank and external own-PNC history;
this is NOT a claim that its proposer/history generator is world-free.
The PDM reference plan/progress is a shared current-input planning prior.
Geometry-relative P16 columns8/9 and all frozen head outputs are excluded.
"""
from revision_v5_twentyfifth_common import *
import numpy as np,argparse,time,copy
from coordiworld.data.openscene import frame_agents,ego_pose,transform_pose,load_log
from revision_v5_twentysecond_kinematics import from_native
from revision_v5_sixteenth_semantics import sat,semantic_features,current_areas
from coordiworld.revision_v5.component_known_b13 import FIELDS
from revision_v5_seventeenth_progress import NAMES as PNAMES
from revision_v5_twentysecond_contact_features import NAMES as CNAMES
P_COLUMNS=[0,1,2,3,4,5,6,7,10,11,12,13,14,15]
FAMILIES=['M12','history','current','spatial']
STATIC_S64=list(range(36,52))
WORLD_S64=list(range(36))+list(range(52,64))
STATIC_C32=[20,21,29,30]
WORLD_C32=[j for j in range(32)if j not in STATIC_C32]
STATIC_V24=list(range(6))+list(range(8,14))+[16]
WORLD_V24=[6,7,14,15]+list(range(17,24))
VERSION='v02_static_shared'

def current_features(a,current,history,families):
    """Explicit allowlist: native ego, submitted ego, current map, observed actors."""
    origin=np.asarray(ego_pose(history[-1]));np.testing.assert_array_equal(origin,np.asarray(current['rear_axle_pose']))
    assert len(history)==4 and int(history[-1]['timestamp'])==current['timestamp_us']
    objects=[frame_agents(h,origin)for h in history];cur=objects[-1];tokens=sorted(cur,key=lambda k:(np.linalg.norm(cur[k][0][:2]),k));M=len(a['valid']);ref=int(a['valid'].argmax())
    native=a['native'];ex=from_native(native,origin)['execution_feature_scene'];trajectory=a['candidates'];parts=[];names=[]
    def add(x,ns):
        x=np.asarray(x,float);assert x.shape==(M,len(ns)),(x.shape,len(ns));parts.append(x);names.extend(ns)
    ts=np.arange(0,41,5)
    ego=np.stack([ex[:,ts,0]/50,ex[:,ts,1]/50,np.sin(ex[:,ts,2]),np.cos(ex[:,ts,2]),native[:,ts,3]/15,native[:,ts,4]/15,native[:,ts,5]/5,native[:,ts,6]/5],-1)
    add(ego.reshape(M,-1),[f'ego_exec_t{t/10:.1f}_{k}'for t in ts for k in ['x50','y50','sin_yaw','cos_yaw','body_vlong15','body_vlat15','body_along5','body_alat5']])
    cand=np.stack([trajectory[:,:,0]/50,trajectory[:,:,1]/50,np.sin(trajectory[:,:,2]),np.cos(trajectory[:,:,2])],-1)
    add(cand.reshape(M,-1),[f'submit_t{(t+1)/2:.1f}_{k}'for t in range(8)for k in ['x50','y50','sin_yaw','cos_yaw']])
    add(a['features198'][:,188:198],list(FIELDS));add(a['R8'],[f'current_road_area_{j}'for j in range(8)])
    # Share every no-future-motion block of the existing representation.
    # Source/availability here is current rail membership/current validity,
    # not a learned future existence/confidence signal. S64's support block
    # uses only this metadata, current map and the submitted ego execution.
    add(np.concatenate([a['features198'][:,:64],a['features198'][:,120:188]],-1),[f'existing_current_geometry_{j}'for j in list(range(64))+list(range(120,188))])
    add(a['x674'][:,np.array(STATIC_S64)+594],[f'S64_current_support_{j}'for j in STATIC_S64])
    add(a['C32'][:,STATIC_C32],[CNAMES[j]for j in STATIC_C32])
    add(a['V24'][:,STATIC_V24],[f'V24_current_static_{j}'for j in STATIC_V24])
    add(a['x674'][:,np.array(P_COLUMNS)+658],[PNAMES[j]for j in P_COLUMNS]);add(a['EC'][:,None],['legal_external_prior_EC'])
    add(np.array([[f==g for g in FAMILIES]for f in families]),['family_'+f for f in FAMILIES])
    # Static current-object contacts along the candidate ego execution. No actor
    # extrapolation is performed; observed velocity remains a separate feature.
    rel=np.zeros((M,4,16));summary=np.zeros((M,4));selected=np.full((M,4),'',dtype='<U64')
    if tokens:
        st=np.stack([cur[k][0]for k in tokens]);typ=np.array([cur[k][1]for k in tokens]);fw=np.stack([np.cos(ex[...,2]),np.sin(ex[...,2])],-1)
        # Standard Pacifica dimensions are the same as the frozen controller.
        eb=np.concatenate([ex[...,:2]+1.461*fw,ex[...,2,None],np.broadcast_to([5.176,2.297],(M,41,2))],-1)
        ab=np.concatenate([st[:,:3],st[:,5:7]],-1);gap=sat(eb[:,:,None],ab[None,None]);minimum=gap.min(1);time_min=gap.argmin(1)
        summary=np.c_[np.clip(minimum.min(1),-50,50)/10,np.log1p((minimum<=0).sum(1)),(gap<=0).any(-1).mean(1),np.clip(gap[:,-1].min(1),-50,50)/10]
        for i in range(M):
            for j,k in enumerate(np.argsort(minimum[i],kind='stable')[:4]):
                t=time_min[i,k];dx,dy=st[k,:2]-ex[i,t,:2];co,si=np.cos(ex[i,t,2]),np.sin(ex[i,t,2]);lon=co*dx+si*dy;lat=-si*dx+co*dy;vyaw=st[k,2]-ex[i,t,2];rv=st[k,3:5]-ex[i,t,3:5]
                rel[i,j]=[np.clip(minimum[i,k],-50,50)/10,t/40,lon/50,lat/50,np.sin(vyaw),np.cos(vyaw),st[k,3]/15,st[k,4]/15,st[k,5]/10,st[k,6]/5,typ[k]/7,1,rv@[co,si]/15,rv@[-si,co]/15,np.clip(gap[i,0,k],-50,50)/10,np.linalg.norm(st[k,:2])/100];selected[i,j]=tokens[k]
    relnames=['min_static_SAT10','closest_time4','long50','lat50','sin_relative_yaw','cos_relative_yaw','observed_vx15','observed_vy15','length10','width5','class7','present','relative_closing15','relative_lateral15','initial_SAT10','initial_distance100']
    add(rel.reshape(M,-1),[f'static_path_actor{j}_{k}'for j in range(4)for k in relnames]);add(summary,['static_min_SAT10','static_contact_count_log','static_contact_time_fraction','static_terminal_SAT10'])
    c=np.concatenate(parts,-1).astype(np.float32);c[~a['valid']]=0
    # Scene context uses all current objects for the count and eight nearest
    # explicit observed states, with four-frame velocity differences when known.
    ctx=[];ctxnames=[]
    for ti,h in enumerate(history):
        p=transform_pose(ego_pose(h),origin);v=np.asarray(h['ego_dynamic_state']);ctx.extend([p[0]/50,p[1]/50,np.sin(p[2]),np.cos(p[2]),*list(v/np.array([15,15,5,5]))]);ctxnames.extend([f'history{ti}_{k}'for k in ['x50','y50','sin_yaw','cos_yaw','vlong15','vlat15','along5','alat5']])
    for j in range(8):
        row=np.zeros(13)
        if j<len(tokens):
            token=tokens[j];st,typ=cur[token];seen=[(float(h['timestamp'])/1e6,o[token][0])for h,o in zip(history,objects)if token in o];acc=(seen[-1][1][3:5]-seen[0][1][3:5])/(seen[-1][0]-seen[0][0])if len(seen)>1 else np.zeros(2)
            row=[st[0]/50,st[1]/50,np.sin(st[2]),np.cos(st[2]),st[3]/15,st[4]/15,st[5]/10,st[6]/5,typ/7,len(seen)/4,acc[0]/5,acc[1]/5,1]
        ctx.extend(row);ctxnames.extend([f'current_nearest{j}_{k}'for k in ['x50','y50','sin_yaw','cos_yaw','vx15','vy15','length10','width5','class7','history_fraction','observed_ax5','observed_ay5','present']])
    ctx.extend([np.log1p(len(tokens)),len(set(objects[0])&set(cur))/max(len(cur),1)]);ctxnames+=['current_object_count_log','history_track_retention']
    result=np.concatenate([c,np.broadcast_to(c[ref],c.shape),c-c[ref],np.broadcast_to(np.array(ctx,np.float32),(M,len(ctx)))],-1).astype(np.float32);result[~a['valid']]=0
    assert np.isfinite(result).all();assert not result[ref,2*c.shape[1]:3*c.shape[1]].any()
    schema=dict(version=VERSION,candidate_columns=names,context_columns=ctxnames,dimensions=result.shape[1],candidate_dim=c.shape[1],layout='candidate, first-valid-reference, exact difference, observed scene context',static_proxy='other actors held at current observed pose; not future-motion forecast',no_predicted_actor_state_access=True,current_support='current rail membership/validity, current map and ego execution; not future probabilities',static_S64=STATIC_S64,static_C32=STATIC_C32,static_V24=STATIC_V24,excluded=['Geometry-relative P16 indices8,9','forecast-dependent S64/C32/V24 columns','dynamic56','all frozen head outputs','Geometry identity'],conditional_scope='common candidate bank and fixed legal prior supplied by upstream world-using policy')
    return result,schema,dict(current_tokens=np.array(tokens),path_current_actor_tokens=selected)

def inputs(a,current,history,families):
    c,schema,trace=current_features(a,current,history,families);ref=int(a['valid'].argmax())
    w=np.concatenate([a['features198'][:,64:120],a['x674'][:,np.array(WORLD_S64)+594],a['C32'][:,WORLD_C32],a['V24'][:,WORLD_V24]],-1).astype(np.float32)
    wx=np.concatenate([w,np.broadcast_to(w[ref],w.shape),w-w[ref]],-1);world=np.concatenate([c,wx],-1);world[~a['valid']]=0
    schema.update(world_addition=['dynamic56: eight predicted-world candidate relation samples','S64 forecast-dependent risk relations/identity','C32 forecast-dependent41-step contact semantics','V24 future displacement/SAT and explicit vulnerable-motion alternative'],world_S64=WORLD_S64,world_C32=WORLD_C32,world_V24=WORLD_V24,world_raw_dim=w.shape[1],world_dimensions=world.shape[1],world_layout='same C plus world candidate/reference/exact difference',C32_names=CNAMES)
    assert np.isfinite(world).all()and not wx[ref,2*w.shape[1]:].any()
    return dict(current=c,world=world),schema,trace

def export(limit=None):
    rows=json.load(open(BASE/'train/manifest.json'))['rows'];rows=rows[:limit]if limit else rows;last=None;t=time.monotonic();count=0;schema=None
    for r in rows:
        d=BASE/'train/scenes'/r['scene_id']
        if not(d/'selection_freeze.json').exists():continue
        if(d/'matched_features_v02_receipt.json').exists():continue
        s=r['source'];path=s['server_path']
        if path!=last:frames=load_log(path);last=path
        j=s['current_index'];h=frames[j-3:j+1];rec=json.load(open(d/'record.json'));a=dict(np.load(d/'features.npz'));cur=json.load(open(d/'current.json'))
        x,schema,trace=inputs(a,cur,h,rec['families']);base=dict(np.load(d/'base_features.npz'));bx,_,_=inputs(base,cur,h,['M12']*12)
        for k in x:np.testing.assert_array_equal(x[k][:12],bx[k])
        # Actual input-isolation check: corrupt all registered world fields;
        # current-only must remain bit-identical, rather than merely zeroing
        # inputs at test time to a model trained with the world.
        if count==0:
            corrupt=copy.deepcopy(a);corrupt['features198'][:,64:120]=123.;corrupt['x674'][:,np.array(WORLD_S64)+594]=-123.;corrupt['C32'][:,WORLD_C32]=42;corrupt['V24'][:,WORLD_V24]=-42
            cc,_,_=current_features(corrupt,cur,h,rec['families']);np.testing.assert_array_equal(cc,x['current'])
            # Recompute shared columns after a genuine forecast perturbation,
            # with current observations, masks, map and ego execution fixed.
            from revision_v5_twentysecond_kinematics import fix_s64
            from revision_v5_twentysecond_contact_features import contact_features
            from revision_v5_eighteenth_source_features import features as vf
            aa=dict(np.load(d/'world_assets.npz'));aa['world']=aa['world'].copy();aa['world'][...,:2]+=np.array([37.,-19.]);area,_=current_areas(cur,aa['execution_full_local'],ROOT/'data/maps');ss,_,tr=semantic_features(aa,area);ss,_=fix_s64(aa,cur,ss,tr);cf,_=contact_features(aa,cur,area);vv,_=vf(aa)
            np.testing.assert_array_equal(ss[:,STATIC_S64],a['x674'][:,np.array(STATIC_S64)+594]);np.testing.assert_array_equal(cf[:,STATIC_C32],a['C32'][:,STATIC_C32]);np.testing.assert_array_equal(vv[:,STATIC_V24],a['V24'][:,STATIC_V24])
        np.savez_compressed(d/'matched_features_v02.npz',**x,**trace);write(d/'matched_features_v02_receipt.json',dict(complete=True,created_utc=now(),source_sha=sha(__file__),input_sha=sha(d/'features.npz'),features_sha=sha(d/'matched_features_v02.npz'),schema=schema,world_perturbation_smoke_executed=count==0,shared_columns_future_perturbation_exact=True if count==0 else None,old_M12_features_exact_within_H3=True));count+=1
    if schema is not None:write(BASE/'train/FEATURE_SCHEMA_v02.json',schema)
    else:schema=json.load(open(BASE/'train/FEATURE_SCHEMA_v02.json'))
    print('R15_FEATURES',VERSION,count,round(time.monotonic()-t,1),schema['dimensions'],schema['world_dimensions'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();export(a.limit)
