"""Conditional slow spatial entry v02: same two-attempt budget, replace the rejected1.2m/s8m option by a tighter6m0.8m/s connection. Original risk gates and world unchanged."""
import importlib.util,time
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_map_supply import local_execution,separation
CONFIG=dict(version='MapSupply_timing_v02_tight',source_type='Parking',profiles=[dict(join_m=8.,cap_mps=.8,curvature_limit=.3),dict(join_m=6.,cap_mps=.8,curvature_limit=.5)],max_additional_attempts=2,trigger='MapSupply triggered, but none of its admitted spatial candidates pass the original risk qualification',risk_relaxation=False,P0=False)
def install(policy,cmap=None):
    import revision_v5_twentyeighth_map_supply as historical
    from h_generic_factory import GenericSupply
    path=Path(historical.__file__);spec=importlib.util.spec_from_file_location('private_map_timing_factory',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CONFIG=module.CONFIG|dict(max_attempts=1);factory=GenericSupply(policy.map,cmap,module=module,offsets=(0.,));old=policy.base_inputs
    def inputs(pipe,h,source):
        from revision_v5_twentythird_factory import ec_matrix
        a,asset,native,current,receipt=old(pipe,h,source);ms=receipt.get('map_supply',{});active=bool(ms.get('trigger')and ms.get('topology',{}).get('source_type')==CONFIG['source_type']and not any(x.get('admitted')and x.get('eligible')for x in ms.get('attempts',[])));r=dict(config=CONFIG,trigger=active,attempts=[]);t=time.perf_counter()
        if active:
            for profile in CONFIG['profiles']:
                cap=profile['cap_mps'];module.CONFIG.update(speed_max=cap,join_lengths_m=[profile['join_m']],curvature_limit_per_m=profile['curvature_limit']);generated,_=factory.generate(h[-1])
                if not generated:continue
                ca,meta,_=generated[0];meta=dict(meta,speed_cap_mps=cap,admitted=False);r['attempts'].append(meta)
                if ca is None:continue
                if any(np.linalg.norm(ca[:,:2]-p[:,:2],axis=1).max()<1e-3 for p in a['candidates'][a['valid']]):meta['reject']='duplicate';continue
                _,ex,_=policy.execution(h,ca[None],np.ones(1,bool));margin,actor=separation(local_execution(ex[0],h[-1]),h[-1]);meta.update(execution_staticSAT_m=margin,execution_actor=actor,executed_forward_m=float(local_execution(ex[0],h[-1])[-1,0]))
                if margin<.1:meta['reject']='execution_current_stationary_box';continue
                a,asset,native,ar=policy.add_candidate(pipe,h,source,a,asset,current,ca);np.testing.assert_allclose(native[-1],ex[0],atol=1e-7,rtol=0);meta.update(admitted=True,index=len(a['valid'])-1,feature_receipt=ar)
        off=None if policy.prior is None else round((h[-1]['timestamp']-policy.prior_ts)/1e5);ec=None if off is None else ec_matrix(native,policy.prior,off);pred=policy.model.predict(a,ec)
        for x in r['attempts']:
            if x['admitted']:
                i=x['index'];x.update(rawNC=float(pred['raw_physical_nc'][i]),rawDAC=float(pred['raw_dac_event'][i]),eligible=bool(pred['eligible'][i]),score=float(pred['final_scores'][i]),EC=None if ec is None else float(ec[i]))
        r.update(choice=int(pred['choice']),selected_slow=any(x.get('index')==int(pred['choice'])for x in r['attempts']if x['admitted']),seconds=time.perf_counter()-t);receipt['map_timing']=r;return a,asset,native,current,receipt
    policy.base_inputs=inputs;return policy
