"""Complete216 analysis under the published plan. No partial score headline or dispatch.

Machine-readable CSV follows the existing scientific ledger format, not a new workbook.
All original attempts/telemetry remain in the indexed physical episode directories.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import json,hashlib,datetime,math,sys,csv
from confirmation_statistics import contrast
from confirmation_motion import motion,summary
EVENTS=['collisions_layout','collisions_pedestrian','collisions_vehicle','red_light','stop_infraction','outside_route_lanes','min_speed_infractions','yield_emergency_vehicle_infractions','scenario_timeouts','route_dev','vehicle_blocked','route_timeout']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p):return [json.loads(line)for line in Path(p).open()]if Path(p).exists()else[]
def prestart_failures(run,f):
    rows=[];freeze_sha=sha(run/'protocol/CONFIRMATION_FREEZE_v01.json')
    for p in sorted((run/'confirmation_v01').glob('operational_recovery_*/BOOTSTRAP_FAILURE_*.json')):
        row=read(p);j=f['schedule'][row['job_index']]
        assert row['freeze_sha256']==freeze_sha and all(row[k]==j[k]for k in ['route','seed','method'])
        assert row['scenario_started']is False and row['policy_decisions']==0 and row['predecision_retry_consumed']==1
        assert sha(row['source'])==row['source_sha256']
        rows.append(dict(row,record_type='BOOTSTRAP_FAILURE',cohort='CONFIRMATION_PRESTART_INFRASTRUCTURE',event_source=str(p),event_source_sha256=sha(p)))
    assert all(n<=f['failures']['predecision_infrastructure_retries']for n in Counter(x['job_index']for x in rows).values())
    return rows
def official_fields(record):
    scores=(record or{}).get('scores',{});events=(record or{}).get('infractions',{});ds=float(scores.get('score_composed',0));rc=float(scores.get('score_route',0));assert math.isfinite(ds)and math.isfinite(rc)
    if record:assert abs(ds-max(0.,rc*float(scores['score_penalty'])))<1e-3
    status=(record or{}).get('status','MISSING_OFFICIAL_RECORD')
    return dict(DS=ds,RC=rc,status=status,official_record_present=record is not None,success=int(bool(record)and status in ['Completed','Perfect']and not any(v for k,v in events.items()if k!='min_speed_infractions')),events={e:len(events[e])if record and e in events else None for e in EVENTS},other_official_events={e:len(v)for e,v in events.items()if e not in EVENTS},official_route_completed_km=float(record['meta']['route_length']*rc/100000)if record else None)
def arm_summary(rows):
    n=len(rows);events={}
    for e in EVENTS:
        known=[r for r in rows if r['events'][e]is not None];with_km=[r for r in known if r['motion']['actual_logged_ego_path_km']is not None];km=sum(r['motion']['actual_logged_ego_path_km']for r in with_km)
        events[e]=dict(known_episodes=len(known),unknown_episodes=n-len(known),known_event_total=sum(r['events'][e]for r in known),all_scheduled_event_total=sum(r['events'][e]for r in known)if len(known)==n else None,distance_matched_episodes=len(with_km),distance_matched_km=km,events_per_actual_km=sum(r['events'][e]for r in with_km)/km if km else None)
    diagnostics={}
    scalar=['stationary_sample_fraction_v_le_0_1'];nested=['speed_mps','moving_speed_mps','lon_accel_abs','lat_accel_abs','lon_jerk_abs','lat_jerk_abs','moving_lon_accel_abs','moving_lat_accel_abs','moving_lon_jerk_abs','moving_lat_jerk_abs']
    for name in scalar+[x+'_p95'for x in nested]:
        route_values=[];excluded=[]
        for route in sorted({r['route']for r in rows}):
            rr=[r for r in rows if r['route']==route];values=[r['motion'].get(name)if name in scalar else r['motion'].get(name[:-4],{}).get('p95')for r in rr]
            if len(rr)==3 and all(v is not None for v in values):route_values.append(sum(values)/3)
            else:excluded.append(route)
        diagnostics[name]=dict(route_equal_mean=sum(route_values)/len(route_values)if route_values else None,complete_routes=len(route_values),excluded_routes=excluded,scope='available complete3seed routes; missing diagnostic routes explicit')
    return dict(episodes=n,routes=len({r['route']for r in rows}),DS_route_equal=sum(r['DS']for r in rows)/n,RC_route_equal=sum(r['RC']for r in rows)/n,success_integer=sum(r['success']for r in rows),success_denominator=n,official_records=sum(r['official_record_present']for r in rows),operational_missing_records=sum(not r['official_record_present']for r in rows),timeout_or_stall_episodes=sum(r['timeout_or_stall_derived']for r in rows),all_attempt_wall_seconds=sum(r['all_attempt_wall_seconds']for r in rows),events=events,motion=diagnostics,status_counts=dict(Counter(r['status']for r in rows)))
def paired_changes(rows,routes,seeds,a,b):
    idx={(r['route'],r['seed'],r['method']):r for r in rows};out={};signs=Counter();event_rows=[]
    for route in routes:
        delta=sum(Decimal(str(idx[(route['id'],s,a)]['DS']))-Decimal(str(idx[(route['id'],s,b)]['DS']))for s in seeds)
        signs['positive'if delta>0 else'negative'if delta<0 else'zero']+=1
    for e in EVENTS:
        known=[];unknown=[]
        for route in routes:
            for seed in seeds:
                av=idx[(route['id'],seed,a)]['events'][e];bv=idx[(route['id'],seed,b)]['events'][e];pair=dict(route=route['id'],seed=seed,arm_a=a,arm_b=b,event=e,a_count=av,b_count=bv,difference=None if av is None or bv is None else av-bv)
                event_rows.append(pair)
                if pair['difference']is None:unknown.append([route['id'],seed])
                else:known.append(pair)
        out[e]=dict(known_pairs=len(known),unknown_pairs=unknown,increased_event_count=sum(max(p['difference'],0)for p in known),reduced_event_count=sum(max(-p['difference'],0)for p in known),newly_penalized_pairs=sum(p['a_count']>0 and p['b_count']==0 for p in known),no_longer_penalized_pairs=sum(p['a_count']==0 and p['b_count']>0 for p in known),interpretation='paired observed counts, not counterfactual identity-matched accidents')
    return dict(route_signs={s:signs[s]for s in ['positive','negative','zero']},events=out),event_rows
def collect(run,f):
    sys.path.insert(0,str(run/'versions/confirmation_v01'));from confirmation_contract import physical_output
    from analyze_development import mechanics,timeout_or_stall
    q=run/'confirmation_v01';freeze_sha=sha(run/'protocol/CONFIRMATION_FREEZE_v01.json');rows=[];attempt_rows=[];inventory=[]
    for j in f['schedule']:
        jr=read(q/'jobs'/f"{j['index']:03d}.json");m=f['methods'][j['method']];fp=Path(jr['final']);a=read(fp)
        assert jr['gate_passed']and jr['freeze_sha256']==freeze_sha==a['freeze_sha256']and a['job']==j
        assert fp.parent.resolve()==physical_output(j,m,f['storage_roots']).resolve()
        attempts=[]
        for k in range(1,a['attempt']+1):
            ap=fp.parent/f'attempt{k}'/'ATTEMPT.json';attempt=read(ap);assert attempt['freeze_sha256']==freeze_sha and attempt['attempt']==k
            if k<a['attempt']:assert not attempt['decision_started']and not attempt['terminal']
            attempts.append(attempt);attempt_rows.append(dict(record_type='ATTEMPT',cohort=a['cohort'],**j,attempt=k,terminal=attempt['terminal'],decision_started=attempt['decision_started'],returncode=attempt['returncode'],wall_timeout=attempt['wall_timeout'],wall_seconds=attempt['wall_seconds'],official_record_present=attempt['official_record']is not None,parse_errors=attempt['parse_errors'],source=str(ap),source_sha256=sha(ap),freeze_sha256=freeze_sha))
        assert 1<=len(attempts)<=2
        ad=fp.parent/f"attempt{a['attempt']}";tr=ad/'trace';traces=jl(tr/'trace.jsonl');metric=jl(tr/'actual_metric_10hz.jsonl');mv=motion(traces,metric,m['kind']);capture=read(tr/'metric_capture.json')if(tr/'metric_capture.json').exists()else{}
        row=dict(record_type='EPISODE',cohort=a['cohort'],**j,actual_method_id=a['method'],track=m['track'],input_permissions=m.get('input_permissions',m.get('identity')),actual_config_sha256=a['config_sha256'],template_config_sha256=a['template_config_sha256'],template_config_hash_encoding=m['config_hash_encoding'],freeze_sha256=freeze_sha,attempt=a['attempt'],decision_started=a['decision_started'],wall_timeout=a['wall_timeout'],returncode=a['returncode'],wall_seconds=a['wall_seconds'],all_attempt_wall_seconds=sum(x['wall_seconds']for x in attempts),**official_fields(a['official_record']),motion=mv,metric_capture_finalized='samples'in capture,metric_capture_errors=capture.get('errors'),mechanics=mechanics(tr)if m['kind']=='internal'else None,prefix_validation=a['prefix_validation'],interface_validation=a['interface_validation'],source=str(fp),source_sha256=sha(fp),physical_directory=str(fp.parent),logical_final=jr['logical_final'])
        stall_input=dict(row,mechanics={**(row['mechanics']or{}),'actual_trace_samples':mv['valid_trace_samples'],'stationary_sample_fraction_v_le_0_1':mv['stationary_sample_fraction_v_le_0_1']})
        row['timeout_or_stall_derived']=timeout_or_stall(stall_input);row['run_step_seconds']=summary([float(t['run_step_seconds'])for t in traces if t.get('run_step_seconds')is not None and math.isfinite(float(t['run_step_seconds']))]);rows.append(row)
        inventory.append(dict(job=j,physical_directory=str(fp.parent),logical_directory=str(Path(jr['logical_final']).parent),final_sha256=sha(fp),attempts=len(attempts)))
    return rows,attempt_rows,inventory
def flattened(record,prefix=''):
    out={}
    for key,value in record.items():
        name=prefix+key
        if isinstance(value,dict):out.update(flattened(value,name+'__'))
        elif isinstance(value,list):out[name]=json.dumps(value,ensure_ascii=False,separators=(',',':'))
        elif value is None:out[name]='UNKNOWN'if name.startswith('events__')or name in ['a_count','b_count','difference']else'MISSING'
        else:out[name]=value
    return out
def main(run):
    q=run/'confirmation_v01';jobs=list((q/'jobs').glob('*.json'));final=read(q/'COORDINATOR_FINAL.json')if(q/'COORDINATOR_FINAL.json').exists()else None
    if len(jobs)!=216 or not final or final['status']!='CONFIRMATION_MATRIX_COMPLETE':print(json.dumps(dict(status='PENDING_COMPLETE_FIXED216',accounted=len(jobs),expected=216,final=final,no_outcomes_read=True)));return
    out=run/'analysis/confirmation';out.mkdir(parents=True,exist_ok=True)
    if(out/'FINAL_v01.json').exists():print(json.dumps(dict(status='EXISTING_COMPLETE_ANALYSIS',path=str(out/'FINAL_v01.json'),sha256=sha(out/'FINAL_v01.json'))));return
    f=read(run/'protocol/CONFIRMATION_FREEZE_v01.json')
    for rel in ['code/confirmation_statistics.py','code/confirmation_motion.py','code/audit_initial_inputs.py']:assert sha(run/rel)==f['file_sha256'][rel]
    rows,attempts,inventory=collect(run,f);assert len(rows)==216 and all(sum(r['method']==m for r in rows)==36 for m in f['methods'])
    prestart=prestart_failures(run,f)
    for row in rows:
        related=[e for e in prestart if e['job_index']==row['index']];row['prestart_infrastructure_failure_count']=len(related);row['prestart_failure_event_sources']=[e['event_source']for e in related]
        assert row['attempt']-1+len(related)<=f['failures']['predecision_infrastructure_retries'],'Predecision retry budget exceeded'
    guards=[read(p)for p in sorted((q/'input_guards').glob('*.json'))];assert len(guards)==72 and all(g['passed']for g in guards)
    arms={label:arm_summary([r for r in rows if r['method']==label])for label in f['methods']};strata={s:{label:arm_summary([r for r in rows if r['method']==label and r['stratum']==s])for label in f['methods']}for s in sorted({r['stratum']for r in f['routes']})};primary={};sensitivity={};changes={};event_rows=[];export=[*rows,*attempts,*prestart]
    for a,b in [('Map-'+f['r_star'],'Generic-'+f['r_star']),('Map-CW','Map-Simple')]:
        name=a+' minus '+b;primary[name]=contrast(rows,f['routes'],f['seeds'],a,b);aligned={(g['route'],g['seed'])for g in guards if g['arms']==[a,b]and g['audit']['normalized_observations_exact_equal']};sensitivity[name]=dict(common_records=contrast(rows,f['routes'],f['seeds'],a,b,common_records=True),aligned_inputs=contrast(rows,f['routes'],f['seeds'],a,b,aligned_pairs=aligned));changes[name],ee=paired_changes(rows,f['routes'],f['seeds'],a,b);event_rows.extend(ee)
        export.append(dict(record_type='PRIMARY_COMPARISON',**primary[name],**changes[name]))
        export.extend(dict(record_type='PRIMARY_ROUTE_EFFECT',direction=name,**r)for r in primary[name]['route_effects'])
        for kind,res in sensitivity[name].items():export.append(dict(record_type='SENSITIVITY',sensitivity=kind,**res))
    export.extend(dict(record_type='ARM_SUMMARY',method=m,**v)for m,v in arms.items())
    export.extend(dict(record_type='STRATUM_SUMMARY',stratum=s,method=m,**v)for s,vv in strata.items()for m,v in vv.items())
    export.extend(dict(record_type='PAIRED_EVENT_COUNTS',**e)for e in event_rows)
    export.extend(dict(record_type='INPUT_PAIR_AUDIT',route=g['route'],seed=g['seed'],arms=g['arms'],passed=g['passed'],exact_input=g['audit']['normalized_observations_exact_equal'],source=str(q/'input_guards'/f"{g['route']}_{g['seed']}_{0 if g['arms'][1].startswith('Generic')else 1}.json"))for g in guards)
    for source,cohort in [('analysis/development/INITIAL24_FINAL_v02.json','DEVELOPMENT_INITIAL24_EXPOSED6'),('analysis/matched/FINAL_v01.json','DEVELOPMENT_MATCHED12_EXPOSED6')]:export.extend(dict(r,record_type='DEVELOPMENT_EPISODE',cohort=cohort)for r in read(run/source)['rows'])
    flat=[flattened(x)for x in export];columns=['record_type','cohort','method','route','seed']+sorted(set().union(*(r.keys()for r in flat))-{'record_type','cohort','method','route','seed'});csvpath=out/'RESULTS_v01.csv'
    with csvpath.open('w',newline='',encoding='utf-8-sig')as stream:writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    with csvpath.open(encoding='utf-8-sig',newline='')as stream:back=list(csv.DictReader(stream))
    assert len(back)==len(flat)and sum(r['record_type']=='EPISODE'for r in back)==216 and sum(r['record_type']=='ATTEMPT'for r in back)==len(attempts)
    (out/'EPISODE_PHYSICAL_INVENTORY_v01.json').write_text(json.dumps(inventory,indent=2)+'\n')
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohort='COMPLETE_FROZEN_CONFIRMATION216',freeze_sha256=sha(run/'protocol/CONFIRMATION_FREEZE_v01.json'),scheduled=216,completed=216,attempts=len(attempts),r_star=f['r_star'],arms=arms,strata=strata,primary=primary,sensitivities=sensitivity,paired_event_changes=changes,input_guards=guards,rows=rows,attempt_rows=attempts,analysis_plan=f['analysis_plan'],source_sha256=sha(Path(__file__)),statistical_source_sha256=sha(run/'code/confirmation_statistics.py'),motion_source_sha256=sha(run/'code/confirmation_motion.py'),csv=dict(path=str(csvpath),sha256=sha(csvpath),records=len(flat),record_type_counts=dict(Counter(r['record_type']for r in export))),development_not_pooled=True,source_scope=f['scope_limit'])
    result.update(prestart_infrastructure_failures=prestart,prestart_failure_count=len(prestart),predecision_retry_budget_verified=True)
    (out/'FINAL_v01.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='COMPLETE',episodes=216,attempts=len(attempts),prestart_failure_count=len(prestart),primary=primary,csv_records=len(flat))))
def self_test():
    missing=official_fields(None);assert missing['DS']==0 and missing['success']==0 and all(v is None for v in missing['events'].values())
    r=dict(scores=dict(score_composed=100,score_route=100,score_penalty=1),status='Completed',infractions={e:[]for e in EVENTS},meta=dict(route_length=100));r['infractions']['min_speed_infractions']=['one'];assert official_fields(r)['success']==1
    r['infractions']['collisions_vehicle']=['one'];assert official_fields(r)['success']==0
    r['scores']['score_composed']=99
    try:official_fields(r)
    except AssertionError:pass
    else:raise AssertionError('score arithmetic mismatch was accepted')
    assert flattened({'events':missing['events']})['events__collisions_vehicle']=='UNKNOWN'
    import tempfile,contextlib,io
    with tempfile.TemporaryDirectory()as tmp:
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):main(Path(tmp))
        assert json.loads(buf.getvalue())['no_outcomes_read']and not(Path(tmp)/'analysis').exists()
    rows=[]
    for route in range(12):
        for seed in [17,23,42]:rows.append(dict(route=str(route),seed=seed,DS=100,RC=100,success=1,official_record_present=True,timeout_or_stall_derived=False,all_attempt_wall_seconds=1,status='Completed',events={e:0 for e in EVENTS},motion=dict(actual_logged_ego_path_km=.1,stationary_sample_fraction_v_le_0_1=0)))
    rows[0].update(DS=0,RC=0,success=0,official_record_present=False,events={e:None for e in EVENTS});rows[0]['motion']['actual_logged_ego_path_km']=0
    rows[1]['events']['collisions_vehicle']=1;rows[1]['motion']['actual_logged_ego_path_km']=0
    s=arm_summary(rows);assert s['episodes']==36 and s['DS_route_equal']==3500/36
    ev=s['events']['collisions_vehicle'];assert ev['known_episodes']==35 and ev['all_scheduled_event_total']is None and abs(ev['events_per_actual_km']-1/3.4)<1e-12
    with tempfile.TemporaryDirectory()as tmp:
        root=Path(tmp);(root/'protocol').mkdir();freeze=root/'protocol/CONFIRMATION_FREEZE_v01.json';freeze.write_text('{}');folder=root/'confirmation_v01/operational_recovery_v01';folder.mkdir(parents=True);log=folder/'failure.log';log.write_text('lock contention before START');j=dict(route='1',seed=42,method='Map-CW');event=dict(job_index=0,**j,freeze_sha256=sha(freeze),scenario_started=False,policy_decisions=0,predecision_retry_consumed=1,source=str(log),source_sha256=sha(log));(folder/'BOOTSTRAP_FAILURE_000.json').write_text(json.dumps(event));records=prestart_failures(root,dict(schedule=[j],failures=dict(predecision_infrastructure_retries=1)));assert len(records)==1 and records[0]['record_type']=='BOOTSTRAP_FAILURE'and'DS'not in records[0]
    return dict(passed=True,checks=['missing scores retained0/eventsUNKNOWN','strict success ignores min-speed only','collision defeats strict success','official DS arithmetic checked','CSV unknown not zero','incomplete216 reads no scores and writes no result','fixed denominator and known-event coverage','zero-distance collision retained in event-rate numerator','pre-START failure retained as separate event without invented driving score'])
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':print(json.dumps(self_test()))
    else:main(Path(sys.argv[1])if len(sys.argv)>1 else Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z'))
