"""Matched current-static4s continuation witness, WITHOUT adding a stop option.

The stop-supply guardv01 selected its newly supplied stop even when the merge
became risk-qualified. This version isolates the witness from that pool change.
Original NC/DAC qualification and scores remain; absent any qualified witness,
the original eligible set is retained. A witness is not a traffic guarantee.
"""
import numpy as np
from shapely.geometry import Point
from revision_v5_twentyeighth_map_supply import MapSupply,local_execution,separation
from revision_v5_twentyeighth_terminal_viability import predicted_frame
from revision_v5_twentyeighth_w4_execution import Executor
CONFIG=dict(version='current_static_continuation_v02_no_stop',source_type='Parking',max_initial_speed=2.,clearance_m=.1,witness_seconds=4.,add_stop=False,original_risk_qualification=True,fallback='original eligible set if no qualified witness')

def install(policy,cmap=None):
    # The witness factory is identical in both arms, regardless of supplied
    # candidate count or slower timing. Keep the frozen8/10/12m fast factory.
    factory=MapSupply(policy.map,cmap);executor=Executor();old_inputs=policy.base_inputs;old_predict=policy.model.predict;cache={}
    def predict(a,ec=None):
        pr=old_predict(a,ec)
        if a is cache.get('array'):
            original=pr['eligible'].copy();mask=original&cache['viable'];fallback=not mask.any()
            if fallback:mask=original
            pr.update(pre_guard_eligible=original,pre_guard_choice=np.array(pr['choice']),terminal_viable=cache['viable'],guard_fallback=np.array(fallback),eligible=mask,choice=int(np.where(mask,pr['final_scores'],-np.inf).argmax()))
        return pr
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
        for ex,(_,d)in zip(executor.execute(np.stack([ca for ca,_ in valid]),init),valid):
            margin,token=separation(local_execution(ex,future),future);rec['execution_checks'].append(dict(join_length_m=d['join_length_m'],minSAT_m=margin,actor=token))
            if margin>=CONFIG['clearance_m']:return True,rec
        return False,rec
    def inputs(pipe,h,source):
        from revision_v5_twentythird_factory import ec_matrix
        a,asset,native,current,receipt=old_inputs(pipe,h,source);now=h[-1];_,top=factory.targets(now);active=top.get('source_type')==CONFIG['source_type']and np.linalg.norm(now['ego_dynamic_state'][:2])<=CONFIG['max_initial_speed'];r=dict(config=CONFIG,active=bool(active),topology=top,checks=[],stop_appended=False)
        if active:
            off=None if policy.prior is None else round((now['timestamp']-policy.prior_ts)/1e5);ec=None if off is None else ec_matrix(native,policy.prior,off);pred=old_predict(a,ec);mask=np.ones(len(a['valid']),bool)
            for i in np.flatnonzero(pred['eligible']):
                ok,detail=witness(now,native[i]);mask[i]=ok;r['checks'].append(dict(index=int(i),viable=bool(ok),detail=detail))
            cache.update(array=a,viable=mask);p=predict(a,ec);r.update(original_choice=int(pred['choice']),new_choice=int(p['choice']),eligible_before=np.flatnonzero(pred['eligible']).tolist(),eligible_after=np.flatnonzero(p['eligible']).tolist(),fallback=bool(p['guard_fallback']))
        else:cache.clear()
        receipt['terminal_viability']=r;return a,asset,native,current,receipt
    policy.base_inputs=inputs;return policy
