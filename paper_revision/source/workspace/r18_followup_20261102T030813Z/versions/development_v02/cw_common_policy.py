"""L1 readout split after a head-independent common pool and physical checks."""
import copy,hashlib,time
from contextlib import contextmanager
import numpy as np
from shapely.geometry import Point
from cw_followup_physics import local_native,cv_box_check,simple_rank
from cw_controller_prefix import roll
from cw_supply_v2 import Factory,CONFIG
from revision_v5_twentyeighth_map_supply import separation

def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

@contextmanager
def independent_readout_initialization(simple):
    """Simple also avoids loading unused CW head checkpoint weights."""
    if not simple:
        yield;return
    import revision_v5_twentythird_policy as parent
    original=parent.PNC
    class NoCWHeads:
        bindings={}
        def predict(self,*args,**kwargs):raise AssertionError('Simple must never use CW heads')
    parent.PNC=NoCWHeads
    try:yield
    finally:parent.PNC=original

def observed_base(pipe,h,source):
    """Original frozen feature view, without the C32-triggered brake replacement.

    World/features remain shared upstream learning. None of these predictions
    controls candidate formation, admission, witness or Simple selection.
    """
    from revision_v5_twentythird_features import fix_s64,contact_features,vf,rf
    b=pipe.inputs(h,source);ss,view=fix_s64(b['assets'],b['current'],b['S64'],b['S64_trace']);x=b['x'][0].copy();x[:,594:658]=ss;cf,_=contact_features(b['assets'],b['current'],b['areas'])
    a=dict(candidates=b['candidates'],valid=b['valid'].copy(),geo=np.array(b['Geometry_HC']),features198=b['features198'],x674=x,old_x674=b['x'][0].copy(),V24=vf(b['assets'])[0],R8=rf(b['areas'],b['valid']),C32=cf)
    assert len(a['valid'])==12 and a['valid'].any()
    return a,b['assets'],view['execution_native'],b['current'],dict(original_C32_brake_replacement_removed=True,upstream_world_learning_retained=True,common_head_predict_calls=0)

class Readout:
    def __init__(self,mode,cw,cache):self.mode=mode;self.cw=cw if mode=='CW' else None;self.cache=cache;self.calls=0
    def predict(self,a,ec=None):
        assert a is self.cache['array'];self.calls+=1
        if self.mode=='CW':
            p=self.cw.predict(a,ec);assert p['eligible'].any() and a['valid'][p['choice']]
            p['head_calls']=np.array(6);p['all_physical_NC_above_half']=np.array(bool((p['raw_physical_nc'][a['valid']]>.5).all()));return p
        forward=local_native(self.cache['native'],self.cache['frame'])[:,-1,0]
        # P16 distance is the frozen geometric projection of native ego motion
        # on the currently supplied navigation/PDM route, not a predicted head.
        progress=np.where(a['x674'][:,668].astype(bool),a['x674'][:,658]*100,forward)
        choice,rows=simple_rank(self.cache['native'],self.cache['frame'],a['valid'],route_progress=progress,stationary_speed=.2)
        order=sorted(range(len(rows)),key=lambda i:tuple(rows[i]['rank_key']));score=np.empty(len(rows));score[order]=np.arange(len(rows))
        return dict(choice=choice,scores=score,final_scores=score,eligible=a['valid'].copy(),head_calls=np.array(0),rule_only=np.array(True))

def install(policy,agent):
    from revision_v5_twentyeighth_terminal_viability import stop_candidate,predicted_frame
    from revision_v5_twentyeighth_w4_execution import Executor
    mode=agent.cfg['supply_mode'];readout=agent.cfg['readout'];assert mode in ['Base','Generic','Map'] and readout in ['CW','Simple']
    factory=Factory(policy.map,agent.hero_actor.get_world().get_map(),agent.supply_physics);executor=Executor();cache={};policy.model=Readout(readout,policy.model,cache)
    def witness(now,native):
        state=native[-1];future=predicted_frame(now,state);future['controller_state']=copy.deepcopy(now['controller_state'])
        future['controller_state']['measured_front_steer_rad']=float(np.arctan(agent.supply_physics['wheelbase_m']*state[9]/max(abs(state[3]),.2)))
        lanes,top=factory.targets(future)
        for lane in lanes:
            q=lane.baseline_path.get_nearest_pose_from_position(Point(state[:2]))
            if lane.polygon.covers(Point(state[:2])) and np.cos(state[2]-q.heading)>.95:return True,dict(reason='terminal_aligned_inside_legal_lane',head_calls=0)
        proposals,top=factory.generate_stage(future,'Map');valid=[(ca,d) for ca,d,_ in proposals if ca is not None]
        rec=dict(reason='continuation_static_witness',topology=top,head_calls=0,attempts=[d for _,d,_ in proposals],execution_checks=[],scope='hypothetical next-plan static continuation using frozen native surrogate; not future CARLA control or dynamic safety')
        if not valid:return False,rec
        init=executor.initial(dict(timestamp_us=future['timestamp'],rear_pose=state[:3].tolist(),rear_velocity_body=state[3:5].tolist(),rear_acceleration_body=state[5:7].tolist(),tire_angle_surrogate=float(state[7]),yaw_rate=float(state[9]),yaw_acceleration=float(state[10])))
        for ex,(_,d) in zip(executor.execute(np.stack([ca for ca,_ in valid]),init),valid):
            sep,actor=separation(local_native(ex,future),future);ok=sep>=CONFIG['clearance_m'];rec['execution_checks'].append(dict(join_length_m=d['join_length_m'],minSAT_m=float(sep),actor=actor,pass_static=bool(ok)))
            if ok:return True,rec
        return False,rec
    def check(now,ca,ex,top,witness_required=True):
        pref,control=roll(now,ca,ex,now['controller_state'],agent.supply_physics);cv=cv_box_check(pref,now,dt=.05,stationary_speed=.2);startup_bracket=[]
        if now['controller_state'].get('actual_control',{}).get('gear',1)==0:
            # Current neutral is observable; future engagement time is not.
            # Check the held-neutral and immediate-engagement predictions plus
            # fixed interpolants. This is a bounded diagnostic bracket, not a
            # certified reachable set or a claim of absolute collision safety.
            engaged,_=roll(now,ca,ex,now['controller_state'],agent.supply_physics,assume_engaged=True)
            for weight in [.25,.5,.75,1.]:
                path=pref+weight*(engaged-pref);cr=cv_box_check(path,now,dt=.05,stationary_speed=.2);startup_bracket.append(dict(engagement_fraction=weight,cv=cr))
            cv=dict(cv,cv_clear=bool(cv['cv_clear'] and all(x['cv']['cv_clear'] for x in startup_bracket)))
        static,actor=separation(local_native(ex,now),now)
        geometry_static,gactor=separation(np.r_[[[0.,0.,0.]],ca],now)
        long_ok=True;wr=None
        if witness_required and top.get('source_type')=='Parking' and np.linalg.norm(now['ego_dynamic_state'][:2])<=2. and cv['cv_clear'] and static>=.1:
            long_ok,wr=witness(now,ex)
        good=bool(cv['cv_clear'] and static>=.1 and geometry_static>=.1 and long_ok)
        reasons=[]
        if not cv['cv_clear']:reasons.append('controller_prefix_dynamic_bbox')
        if static<.1:reasons.append('native_4s_static_bbox')
        if geometry_static<.1:reasons.append('candidate_geometry_static_bbox')
        if not long_ok:reasons.append('no_static_continuation_witness')
        return good,dict(admitted_physics=good,reject_reasons=reasons,controller_prefix_cv=cv,startup_gear_bracket=startup_bracket,native_static_minSAT_m=float(static),native_static_actor=actor,geometry_static_minSAT_m=float(geometry_static),geometry_static_actor=gactor,continuation=wr,prefix_model=control),pref
    def inputs(pipe,h,source):
        t=time.perf_counter();a,asset,native,current,receipt=observed_base(pipe,h,source);now=h[-1];lanes,top=factory.targets(now);raw_count=len(a['valid']);original_valid=a['valid'].copy()
        # First inspect the original 12 for an actual predicted deceleration.
        v0=max(0.,float(now['ego_dynamic_state'][0]));decel=[int(i) for i in np.flatnonzero(a['valid']) if native[i,-1,3]<=max(.1,.5*v0) and float(np.mean(native[i,:,3]))<=v0+.1]
        appended_stop=False;stop_index=None
        if not decel:
            stop=stop_candidate(now);a,asset,native,ar=policy.add_candidate(pipe,h,source,a,asset,current,stop);stop_index=len(a['valid'])-1;appended_stop=True
        else:stop_index=min(decel,key=lambda i:(float(np.mean(native[i,:,3])),i));ar=None
        core_count=len(a['valid']);checks=[];prefix=[];feasible=[]
        for i in range(core_count):
            if not a['valid'][i]:checks.append(dict(admitted_physics=False,reject_reasons=['original_geometry_invalid']));prefix.append(np.zeros((11,3)));feasible.append(False);continue
            ok,rec,pr=check(now,a['candidates'][i],native[i],top,witness_required=i!=stop_index);checks.append(rec);prefix.append(pr);feasible.append(ok)
        # A physically admissible shared stop competes in both readouts.
        # It still cannot count as positive-progress coverage.
        covered=set()
        for i in range(raw_count):
            if feasible[i] and local_native(native[i],now)[-1,0]>=2.:covered.update(factory.terminal(native[i,-1,:3],lanes))
        trigger=not bool(covered) and np.linalg.norm(now['ego_dynamic_state'][:2])<=2.
        attempts=[]
        def append_stage(slow):
            nonlocal a,asset,native
            props,_=factory.generate_stage(now,mode,covered,slow=slow)
            admitted=0
            for ca,rec,_ in props:
                rec['admitted']=False
                if ca is None:attempts.append(rec);continue
                rec['candidate_sha']=digest(ca)
                duplicate=any(np.linalg.norm(ca[:,:2]-old[:,:2],axis=1).max()<1e-3 and abs(np.arctan2(np.sin(ca[:,2]-old[:,2]),np.cos(ca[:,2]-old[:,2]))).max()<1e-4 for old in a['candidates'][a['valid']])
                if duplicate:rec['reject']='duplicate_existing_candidate';attempts.append(rec);continue
                _,fn,_=policy.execution(h,ca[None],np.ones(1,bool));ok,rr,pr=check(now,ca,fn[0],top)
                rec['execution_check']=rr
                if not ok:rec['reject']='execution_or_continuation';attempts.append(rec);continue
                a,asset,native,feature_receipt=policy.add_candidate(pipe,h,source,a,asset,current,ca);np.testing.assert_allclose(native[-1],fn[0],atol=1e-7,rtol=0)
                feasible.append(True);checks.append(rr);prefix.append(pr);rec.update(admitted=True,candidate_index=len(a['valid'])-1,feature_receipt=feature_receipt,native_surrogate_terminal_legal_lanes=factory.terminal(native[-1,-1,:3],lanes));attempts.append(rec);admitted+=1
            return admitted
        if trigger and mode!='Base':
            admitted=append_stage(False)
            if top.get('source_type')=='Parking' and admitted==0:append_stage(True)
        assert len(attempts)<=5 and len(a['valid'])<=18 and len(feasible)==len(a['valid'])
        raw_valid=a['valid'].copy();physical=np.asarray(feasible,bool)&raw_valid;no_feasible=not physical.any()
        if no_feasible:
            # One shared deterministic emergency action, explicitly uncertified
            # if its prefix/static checks fail. Never call relative CW risk safe.
            physical[stop_index]=True
        a['valid']=physical
        old_geo=int(a['geo'])
        if not physical[old_geo]:
            from coordiworld.revision_v5.component_known_b13 import geometry_known_score
            f=a['features198'];g=int(np.where(physical,geometry_known_score(f[None],f[None,:,188:],physical[None])[0],-np.inf).argmax());a['geo']=np.array(g)
            for key in ['x674','old_x674']:
                p=a[key][:,658:];p[:,8]=p[:,0]-p[g,0];p[:,9]=p[:,4]-p[g,4]
        a['common_raw_valid']=raw_valid;a['controller_prefix_local']=np.stack(prefix);a['common_physical_mask']=np.asarray(feasible,bool);a['common_emergency']=np.array(no_feasible)
        coverage_after=set()
        for i in np.flatnonzero(np.asarray(feasible,bool)&raw_valid):
            if local_native(native[i],now)[-1,0]>=CONFIG['positive_progress_m']:coverage_after.update(factory.terminal(native[i,-1,:3],lanes))
        cache.update(array=a,native=native,frame=now)
        receipt['supply_v2']=dict(config=CONFIG,supply_mode=mode,readout=readout,interface='Supply_v2_L1_shared_brake_v01',common_head_predict_calls=0,topology=top,covered_legal_lanes=sorted(covered),trigger=bool(trigger),attempts=attempts,physical_checks=checks,core_count=core_count,base_original_count=raw_count,max_extra_attempts=5,existing_deceleration_indices=decel,shared_stop_appended=appended_stop,stop_index=stop_index,stop_feature_receipt=ar,emergency_only_new_stop=False,no_feasible_action=no_feasible,emergency_brake_prefix_clear=bool(checks[stop_index].get('controller_prefix_cv',{}).get('cv_clear',False)),absolute_safety_claim=False,common_valid_indices=np.flatnonzero(physical).tolist(),pool_sha=digest(a['candidates']),native_sha=digest(native),common_mask_sha=digest(physical),original_geometry_reference=old_geo,physical_geometry_reference=int(a['geo']),seconds=time.perf_counter()-t)
        receipt['supply_v2']['native_surrogate_legal_lane_coverage_after']=sorted(coverage_after)
        return a,asset,native,current,receipt
    policy.base_inputs=inputs
    return policy
