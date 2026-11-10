"""TRAIN-free parking-entry hypothesis: avoid a statically blocked next plan.

This is an additional, separately controlled guard, not MapSupply's original
effect. All original PNC risk gates remain. A shared jerk-bounded stop option
is offered to both arms. Continuation witnesses use predicted ego state and
currently stationary boxes only; no future CARLA observations or labels.
"""
import copy,json
import numpy as np
from shapely.geometry import Point
from revision_v5_twentyeighth_map_supply import MapSupply,rotation,separation,local_execution
from revision_v5_twentyeighth_w4_execution import Executor
CONFIG=dict(version='StaticTerminalViability_v01',applies_to_current_lane_type='Parking',max_initial_speed=2.,stationary_actor_speed=.2,clearance_m=.10,stop_deceleration=3.,stop_jerk=3.,witnesses_max=3,unchanged_risk_qualification=True)

def stop_candidate(h):
    v=max(0.,float(h['ego_dynamic_state'][0]));a=float(h['ego_dynamic_state'][2]);k=float(h['omega']/v)if v>.2 else 0.;p=np.zeros(3);rows=[p.copy()]
    for _ in range(80):
        target=-min(CONFIG['stop_deceleration'],2*v);a+=np.clip(target-a,-CONFIG['stop_jerk']*.05,CONFIG['stop_jerk']*.05);new=max(0.,v+a*.05);distance=(v+new)*.025;mid=p[2]+k*distance/2;p[:2]+=distance*np.array([np.cos(mid),np.sin(mid)]);p[2]+=k*distance;v=new
        if v==0:a=0.
        rows.append(p.copy())
    return np.asarray(rows)[10::10].astype(np.float32)

def predicted_frame(h,state):
    f=copy.deepcopy(h);o=np.array(h['ego2global_translation'][:2]);yaw=h['body_pose_math'][2];origin=state[:2];heading=state[2];R=rotation(yaw);RN=rotation(heading)
    b=np.asarray(h['anns']['gt_boxes'],float).reshape(-1,7);v=np.asarray(h['anns']['gt_velocity_3d'],float).reshape(-1,3);ids=np.flatnonzero(np.linalg.norm(v[:,:2],axis=1)<=CONFIG['stationary_actor_speed']);bb=b[ids].copy();bb[:,:2]=(bb[:,:2]@R.T+o-origin)@RN;bb[:,6]=np.arctan2(np.sin(bb[:,6]+yaw-heading),np.cos(bb[:,6]+yaw-heading));vv=v[ids].copy();vv[:,:2]=vv[:,:2]@R.T@RN
    an={k:np.asarray(val)[ids].tolist()for k,val in h['anns'].items()};an['gt_boxes']=bb.tolist();an['gt_velocity_3d']=vv.tolist();f['anns']=an
    f['ego2global_translation']=[*origin,h['ego2global_translation'][2]];f['ego2global_rotation']=[np.cos(heading/2),0.,0.,np.sin(heading/2)];f['ego_dynamic_state']=state[3:7].tolist();f['omega']=float(state[9]);f['timestamp']=int(h['timestamp'])+4000000;f['sim_seconds']=h['sim_seconds']+4;f['body_pose_math']=[*(origin-RN@np.asarray(h['rear_geometry']['rear_offset_math'])),float(heading)];f['current_only']=False;f['predicted_terminal_view']=True
    return f

def install(policy,cmap=None):
    factory=getattr(policy,'map_supply_factory',None)or MapSupply(policy.map,cmap);executor=Executor();old_inputs=policy.base_inputs;old_predict=policy.model.predict;cache={}
    def predict(a,ec=None):
        pred=old_predict(a,ec)
        if a is cache.get('array'):
            original=pred['eligible'].copy();eligible=original&cache['viable']
            fallback=not eligible.any()
            if fallback:eligible=original
            pred.update(pre_guard_eligible=original,pre_guard_choice=np.array(pred['choice']),terminal_viable=cache['viable'],guard_fallback=np.array(fallback),eligible=eligible,choice=int(np.where(eligible,pred['final_scores'],-np.inf).argmax()))
        return pred
    policy.model.predict=predict
    def witness(h,native):
        state=native[-1];future=predicted_frame(h,state);lanes,top=factory.targets(future)
        if not lanes:return False,dict(reason='no_terminal_connected_lane',topology=top)
        for lane in lanes:
            pt=Point(state[:2]);near=lane.baseline_path.get_nearest_pose_from_position(pt)
            if lane.polygon.covers(pt)and np.cos(state[2]-near.heading)>.95:return True,dict(reason='terminal_aligned_inside_driving_lane')
        proposals,top=factory.generate(future);valid=[(ca,d)for ca,d,_ in proposals if ca is not None];rec=dict(topology=top,geometric_attempts=[d for _,d,_ in proposals],execution_checks=[])
        if not valid:return False,rec
        init=executor.initial(dict(timestamp_us=future['timestamp'],rear_pose=state[:3].tolist(),rear_velocity_body=state[3:5].tolist(),rear_acceleration_body=state[5:7].tolist(),tire_angle_surrogate=float(state[7]),yaw_rate=float(state[9]),yaw_acceleration=float(state[10])))
        executions=executor.execute(np.stack([ca for ca,_ in valid]),init)
        for ex,(_,d)in zip(executions,valid):
            margin,token=separation(local_execution(ex,future),future);rec['execution_checks'].append(dict(join_length_m=d['join_length_m'],minSAT_m=margin,actor=token))
            if margin>=CONFIG['clearance_m']:return True,rec
        return False,rec
    def inputs(pipe,h,source):
        from revision_v5_twentythird_factory import ec_matrix
        a,assets,native,current,receipt=old_inputs(pipe,h,source);now=h[-1];_,top=factory.targets(now);active=top.get('source_type')=='Parking'and np.linalg.norm(now['ego_dynamic_state'][:2])<=CONFIG['max_initial_speed'];r=dict(config=CONFIG,active=bool(active),topology=top,checks=[])
        if active:
            stop=stop_candidate(now);dups=[i for i in np.flatnonzero(a['valid'])if np.linalg.norm(a['candidates'][i,:,:2]-stop[:,:2],axis=1).max()<1e-3]
            if dups:r['stop_index']=int(dups[0]);r['stop_appended']=False
            else:
                a,assets,native,ar=policy.add_candidate(pipe,h,source,a,assets,current,stop);r.update(stop_index=len(a['valid'])-1,stop_appended=True,stop_receipt=ar)
            ts=int(now['timestamp']);off=None if policy.prior is None else round((ts-policy.prior_ts)/1e5);ec=None if off is None else ec_matrix(native,policy.prior,off);pred=old_predict(a,ec);mask=np.ones(len(a['valid']),bool)
            for i in np.flatnonzero(pred['eligible']):
                ok,detail=witness(now,native[i]);mask[i]=ok;r['checks'].append(dict(index=int(i),viable=bool(ok),detail=detail))
            cache.update(array=a,viable=mask);guarded=predict(a,ec);r.update(original_choice=int(pred['choice']),new_choice=int(guarded['choice']),eligible_before=np.flatnonzero(pred['eligible']).tolist(),eligible_after=np.flatnonzero(guarded['eligible']).tolist(),fallback=bool(guarded['guard_fallback']))
        else:cache.clear()
        receipt['terminal_viability']=r;return a,assets,native,current,receipt
    policy.base_inputs=inputs;return policy
