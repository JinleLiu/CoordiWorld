"""Install spatial supply after base_inputs; frozen PNC decision stays intact."""
import numpy as np
from revision_v5_twentyeighth_map_supply import MapSupply,CONFIG,coverage,local_execution,separation

def install(policy,cmap=None):
    assert policy.method=='BaseEC'
    factory=MapSupply(policy.map,cmap);original=policy.base_inputs
    def inputs(pipe,h,source):
        from revision_v5_twentythird_factory import ec_matrix
        a,asset,native,current,receipt=original(pipe,h,source);now=h[-1]
        ts=int(now['timestamp']);off=None if policy.prior is None else round((ts-policy.prior_ts)/1e5)
        ec=None if off is None else ec_matrix(native,policy.prior,off)
        target_lanes,topology=factory.targets(now)
        old=policy.model.predict(a,ec);safe,sat=coverage(a,native,old,now,target_lanes)
        speed=float(np.linalg.norm(now['ego_dynamic_state'][:2]));ago=[f for f in h if (ts-f['timestamp'])>=CONFIG['stalled_history_s']*1e6]
        stalled=bool(ago and speed<CONFIG['stalled_speed_mps']and np.linalg.norm(np.asarray(now['ego2global_translation'][:2])-ago[-1]['ego2global_translation'][:2])<CONFIG['stalled_distance_m'])
        trigger=bool(not safe.any()and(speed<CONFIG['slow_speed_mps']or stalled))
        r=dict(config=CONFIG,trigger=trigger,stalled=stalled,observed_speed_mps=speed,old_safe_positive_indices=np.flatnonzero(safe).tolist(),old_eligible_indices=np.flatnonzero(old['eligible']).tolist(),old_choice=int(old['choice']),attempts=[],topology=topology,coverage_definition='eligible, actual static box clear, forward>=2m, terminal rear axle inside topologically allowed same-direction Driving lane')
        if trigger:
            supplied,topology=factory.generate(now);r['topology']=topology
            for ca,detail,_ in supplied:
                detail=dict(detail,admitted=False);r['attempts'].append(detail)
                if ca is None:continue
                duplicate=any(np.linalg.norm(ca[:,:2]-p[:,:2],axis=1).max()<1e-3 for p in a['candidates'][a['valid']])
                detail['duplicate']=duplicate
                if duplicate:continue
                _,ex,_=policy.execution(h,ca[None],np.ones(1,bool));s,t=separation(local_execution(ex[0],now),now)
                detail.update(execution_current_box_minSAT_m=float(s),execution_nearest_static_actor=t)
                if s<CONFIG['current_box_clearance_m']:detail['reject']='executed_current_stationary_box';continue
                a,asset,native,ar=policy.add_candidate(pipe,h,source,a,asset,current,ca)
                np.testing.assert_allclose(ex[0],native[-1],atol=1e-7,rtol=0)
                detail.update(admitted=True,index=len(a['valid'])-1,feature_receipt=ar)
        ec=None if off is None else ec_matrix(native,policy.prior,off);pr=policy.model.predict(a,ec);safe,sat=coverage(a,native,pr,now,target_lanes)
        r.update(new_safe_positive_indices=np.flatnonzero(safe).tolist(),new_choice=int(pr['choice']),selected_extra=bool(int(pr['choice'])>=12),new_eligible_indices=np.flatnonzero(pr['eligible']).tolist())
        for detail in r['attempts']:
            if detail['admitted']:
                j=detail['index'];detail.update(rawNC=float(pr['raw_physical_nc'][j]),rawDAC=float(pr['raw_dac_event'][j]),eligible=bool(pr['eligible'][j]),score=float(pr['final_scores'][j]),EC=None if ec is None else float(ec[j]))
        receipt['map_supply']=r;return a,asset,native,current,receipt
    policy.base_inputs=inputs;policy.map_supply_factory=factory;return policy
