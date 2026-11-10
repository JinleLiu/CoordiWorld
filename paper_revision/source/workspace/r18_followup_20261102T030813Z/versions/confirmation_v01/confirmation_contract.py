"""Pure six-arm confirmation accounting. No scheduler, simulator or model imports."""
from pathlib import Path
import json,math,hashlib

INTERNAL_LABELS={'Base-CW','Generic-CW','Map-CW','Base-Simple','Generic-Simple','Map-Simple'}
PUBLIC_IDS={'World4Scorer_route_tp','TFpp_single'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def config_digest(config,encoding):
    assert encoding in ['compact_sorted_json_utf8','default_sorted_json_utf8']
    opts={'separators':(',',':')}if encoding=='compact_sorted_json_utf8'else{}
    return hashlib.sha256(json.dumps(config,sort_keys=True,**opts).encode()).hexdigest()
def selected_labels(rstar):
    assert rstar in ['CW','Simple']
    other='Simple'if rstar=='CW'else'CW'
    return ['Base-'+rstar,'Generic-'+rstar,'Map-'+rstar,'Map-'+other,'W4-native','TFpp-single']

def make_schedule(routes,seeds,rstar,storage_roots):
    labels=selected_labels(rstar);jobs=[]
    for ri,r in enumerate(routes):
        for si,seed in enumerate(seeds):
            volume='data'if(ri+si)%3==0 else'root'
            block=str(Path(storage_roots[volume])/f'seed{seed}'/str(r['id']))
            shift=(ri+2*si)%6;order=labels[shift:]+labels[:shift]
            for label in order:
                jobs.append(dict(index=len(jobs),route=str(r['id']),seed=seed,method=label,route_ordinal=ri,seed_ordinal=si,stratum=r['stratum'],slot=(labels.index(label)+ri+si)%3,volume=volume,physical_block=block))
    return jobs

def validate_matrix(f):
    assert f['phase']=='CONFIRMATION'and f['confirmation_authorized_by_this_file']
    assert f['T0_utc']=='2026-11-02T03:08:13+00:00'
    assert f['episode_budget']==216 and len(f['routes'])==12 and f['seeds']==[17,23,42]
    labels=selected_labels(f['r_star']);assert list(f['methods'])==labels
    assert f['primary_directions']==['Map-r* minus Generic-r*','Map-CW minus Map-Simple']
    assert f['interface']=='L1'and f['new_training']is False
    assert len({str(r['id'])for r in f['routes']})==12
    assert not({str(r['id'])for r in f['routes']}&set(f['previously_exposed_routes']))
    from collections import Counter
    assert sorted(Counter(r['stratum']for r in f['routes']).values())==[4,4,4]
    expected=make_schedule(f['routes'],f['seeds'],f['r_star'],f['storage_roots']);assert f['schedule']==expected
    assert len({(j['route'],j['seed'],j['method'])for j in expected})==216
    assert Counter(j['slot']for j in expected)=={0:72,1:72,2:72}
    assert Counter(j['volume']for j in expected)=={'data':72,'root':144}
    for label,m in f['methods'].items():
        assert config_digest(m['config'],m['config_hash_encoding'])==m['config_sha256']
        if label in INTERNAL_LABELS:
            assert m['kind']=='internal'and m['track']=='MAP'
            supply,readout=label.split('-');assert m['config']['supply_mode']==supply and m['config']['readout']==readout
        else:
            assert m['kind']=='native_public'and m['track']=='SENSORS'and m['id']in PUBLIC_IDS
            if m['id']=='World4Scorer_route_tp':assert m['expected_metadata']==dict(method='World4Scorer_route_tp',official_commit='7814310e0bd65c6e21f06b81eedfec2ac9a06aa9',checkpoint_sha256=m['checkpoint_sha'])
            else:assert m['expected_metadata']==dict(method='TFpp_all_towns_seed0',official_commit='48e6a6dbcb133220333bd65e242fbc2b492bb15b',native_model_count=1)
    assert {f['methods'][x]['id']for x in ['W4-native','TFpp-single']}==PUBLIC_IDS
    failures=f['failures'];assert failures['predecision_infrastructure_retries']==1 and failures['after_decision_retries']==0
    assert failures['missing_official_score']==0 and failures['missing_official_events']=='UNKNOWN'
    assert failures['episode_wall_seconds']==3600 and failures['RPC_timeout_seconds']==600
    return True

def decision_steps(traces,kind):
    # Preserve the registered R18/v02 convention, including native startup callbacks.
    if kind=='internal':return sum(int(x.get('policy_replans',0))>0 for x in traces)
    assert kind=='native_public';return len(traces)

def terminal_attempt(record,steps,attempt):
    assert attempt in [1,2]
    return bool(record and record.get('status')=='Completed')or steps>0 or attempt==2

def native_trace_validation(traces,metadata):
    errors=[]
    if not traces:errors.append('no_native_control_callback')
    if not metadata:errors.append('missing_native_setup_metadata')
    last=None
    for i,r in enumerate(traces):
        try:
            t=float(r['sim_seconds']);c=r['control'];v=r['velocity_world']
            assert math.isfinite(t)and(last is None or t>last);last=t
            assert len(v)==3 and all(math.isfinite(float(x))for x in v)
            assert all(math.isfinite(float(c[k]))for k in ['steer','throttle','brake'])
            assert -1.000001<=c['steer']<=1.000001 and 0<=c['throttle']<=1.000001 and 0<=c['brake']<=1.000001
            if'pose'in r:assert len(r['pose'])>=3 and all(math.isfinite(float(x))for x in r['pose'])
            else:assert all(math.isfinite(float(r['ego_world'][k]))for k in ['x','y','z','yaw'])
        except (KeyError,TypeError,ValueError,AssertionError):errors.append('invalid_native_trace_row_'+str(i))
    return dict(passed=not errors,errors=errors,kind='native_original_controller_observation_contract',samples=len(traces),CW_controller_prefix='NOT_APPLICABLE: native public controller retained; no CW rollout substituted',scope='Finite ordered recorded native controls and ego telemetry; no absolute safety or CW prefix-accuracy claim')

def internal_identity_validation(trace,method):
    errors=[]
    try:
        meta=read(Path(trace)/'agent_metadata.json');model=read(Path(trace)/'policy/model_identity.json');cfg=method['config']
        assert meta['method']==method['id']and meta['supply_mode']==cfg['supply_mode']and meta['readout']==cfg['readout']
        simple=cfg['readout']=='Simple'
        assert model['readout']==cfg['readout']and model['CW_head_checkpoint_weights_loaded']==(not simple)and model['CW_head_predictions_used']==(not simple)
        assert meta['actual_controller_unchanged']and model['upstream_frozen_world_retained']
    except (FileNotFoundError,KeyError,ValueError,AssertionError)as e:errors.append(type(e).__name__+':'+str(e))
    return dict(passed=not errors,errors=errors,kind='frozen_internal_interface_identity')

def native_identity_validation(metadata,method):
    errors=[]
    try:
        expected=method['expected_metadata'];assert all(metadata[k]==v for k,v in expected.items())
        config=dict(metadata['config'])
        for k in ['output_dir','eval_seed','method_config_sha']:config.pop(k)
        assert config==method['config']
        assert metadata['config']['method_config_sha']==method['config_sha256']
    except (KeyError,TypeError,AssertionError)as e:errors.append(type(e).__name__+':'+str(e))
    return dict(passed=not errors,errors=errors,kind='registered_native_identity_and_config')

def next_job(schedule,done,running_indices,slot):
    return next((j['index']for j in schedule if j['slot']==slot and j['index']not in done and j['index']not in running_indices),None)

def physical_output(job,method,roots):
    root=Path(roots[job['volume']]);block=root/f"seed{job['seed']}"/job['route'];assert str(block)==job['physical_block']
    assert method['id']not in ['.','..']and'/'not in method['id']
    return block/method['id']
