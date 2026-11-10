"""Read accounted FINALs only; no partial-cohort headline or automatic adoption.

Official event/success semantics follow the final R18 results.py. Missing
scheduled cells remain PENDING, exhausted failures stay in the denominator.
No simulator, checkpoint update, selection-rule mutation or queue dispatch.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,sys,math
import numpy as np
EVENTS=['collisions_layout','collisions_pedestrian','collisions_vehicle','red_light','stop_infraction','outside_route_lanes','min_speed_infractions','yield_emergency_vehicle_infractions','scenario_timeouts','route_dev','vehicle_blocked','route_timeout']
RULE=['red_light','stop_infraction','outside_route_lanes'];STALL=['scenario_timeouts','vehicle_blocked','route_timeout']
RUN=Path(sys.argv[1])if len(sys.argv)>1 else Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quant(x,p):return float(np.quantile(x,p))if len(x)else None
def jsonl(p):return [json.loads(s)for s in p.read_text().splitlines()]if p.exists()else[]
def mechanics(trace):
    plans=jsonl(trace/'policy/plans.jsonl');tr=jsonl(trace/'trace.jsonl');attempts=Counter();rejections=Counter();choices=Counter();chosen_supply=0;chosen_stop=0;triggers=0;no_feasible=0;before=0;after=0;new_covered=0;selected_CW_high_risk=0;cw_known=0;prefix_cv_conflict=0
    for p in plans:
        s=p['base_receipt']['supply_v2'];choices[str(p['choice'])]+=1;triggers+=int(s['trigger']);no_feasible+=int(s['no_feasible_action']);bs=set(s['covered_legal_lanes']);af=set(s.get('native_surrogate_legal_lane_coverage_after',[]));before+=len(bs);after+=len(af);new_covered+=len(af-bs)
        chosen_stop+=int(p['choice']==s['stop_index']);chosen_supply+=int(any(x.get('admitted')and x.get('candidate_index')==p['choice']for x in s['attempts']))
        for a in s['attempts']:
            attempts['attempted']+=1;attempts['admitted']+=int(a['admitted']);attempts['rejected']+=int(not a['admitted'])
            if not a['admitted']:rejections[a.get('reject','unspecified')]+=1
        check=s['physical_checks'][p['choice']];prefix_cv_conflict+=int(not check.get('controller_prefix_cv',{}).get('cv_clear',False))
        npz=trace/'policy'/f"plan_{p['replan']:05d}.npz"
        with np.load(npz,allow_pickle=False)as a:
            if 'PNC_raw_physical_nc'in a:cw_known+=1;selected_CW_high_risk+=int(a['PNC_raw_physical_nc'][p['choice']]>.5)
    xy=np.asarray([r['body_pose'][:2]for r in tr],float);speeds=np.asarray([np.linalg.norm(r['velocity_world'][:2])for r in tr],float)
    duration=float(tr[-1]['sim_seconds']-tr[0]['sim_seconds'])if len(tr)>1 else None
    return dict(replans=len(plans),choices=dict(choices),supply=dict(attempts),supply_rejections=dict(rejections),trigger_replans=triggers,selected_supply_replans=chosen_supply,selected_shared_or_existing_stop_replans=chosen_stop,no_physical_feasible_replans=no_feasible,selected_prefix_CV_conflict_replans=prefix_cv_conflict,native_surrogate_covered_lane_replan_pairs_before=before,native_surrogate_covered_lane_replan_pairs_after=after,native_surrogate_new_covered_lane_replan_pairs=new_covered,actual_trace_samples=len(tr),actual_20Hz_body_path_km=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()/1000)if len(xy)>1 else None,actual_trace_duration_s=duration,stationary_sample_fraction_v_le_0_1=float(np.mean(speeds<=.1))if len(speeds)else None,replan_seconds_p50=quant([p['seconds']for p in plans],.5),replan_seconds_p95=quant([p['seconds']for p in plans],.95),selected_CW_predicted_high_risk_replans=selected_CW_high_risk if cw_known else None,CW_risk_known_replans=cw_known,scope='Coverage is a native surrogate endpoint diagnostic, not actual lane coverage or counterfactual safety; actual motion reported separately. All samples retained.')
def timeout_or_stall(r):
    # TickRuntime is carried by official status, even when event lists are empty.
    st=r['status'].lower();m=r.get('mechanics')or{}
    return bool(r.get('wall_timeout')or any(t in st for t in ['timeout','tickruntime','blocked','stalled'])or any(r['events'].get(e)for e in STALL)or (r['RC']<99.9999 and m.get('actual_trace_samples',0)>=20 and (m.get('stationary_sample_fraction_v_le_0_1')or 0)>=.99))
def main():
    active=read(RUN/'runtime/DEVELOPMENT_ACTIVE.json')if(RUN/'runtime/DEVELOPMENT_ACTIVE.json').exists()else{}
    freeze=RUN/active.get('freeze','protocol/DEVELOPMENT_FREEZE_v01.json');f=read(freeze);q=RUN/active.get('queue_directory','development_v01');rows=[]
    for index,j in enumerate(f['schedule']):
        jr=read(q/'jobs'/f'{index:02d}.json')if(q/'jobs'/f'{index:02d}.json').exists()else{}
        p=q/'episodes'/f"seed{j['seed']}"/j['route']/f['methods'][j['method']]['id']/'FINAL.json'
        if jr.get('final'):p=Path(jr['final'])
        if not p.exists():rows.append(dict(**j,status='PENDING',DS=None,source=str(p)));continue
        a=read(p);reused=jr.get('imported_equivalent_repair',False)
        if reused:assert a['freeze_sha256']==f['parent_freeze_sha256']and sha(p)==jr['original_final_sha256']and jr['gate_passed']and jr['compatibility_audit_sha256']==f['compatibility_audit_sha256']
        else:assert a['freeze_sha256']==sha(freeze)
        r=a['official_record'];sc=(r or{}).get('scores',{});ev=(r or{}).get('infractions',{});status=(r or{}).get('status','MISSING_OFFICIAL_RECORD');ds=float(sc.get('score_composed',0));rc=float(sc.get('score_route',0));assert math.isfinite(ds)and math.isfinite(rc)
        if r:assert abs(ds-max(0.,rc*float(sc['score_penalty'])))<1e-3,('Official score arithmetic',j)
        ad=p.parent/f"attempt{a['attempt']}";trace=ad/'trace';events={e:len(ev[e])if r and e in ev else None for e in EVENTS}
        rows.append(dict(**j,actual_method_id=a['method'],actual_code_directory=a['code_directory'],actual_config_sha256=a['config_sha256'],actual_cohort=a['cohort'],status=status,DS=ds,RC=rc,success=int(bool(r)and status in ['Completed','Perfect']and not any(v for k,v in ev.items()if k!='min_speed_infractions')),events=events,event_complete=all(v is not None for v in events.values()),official_record_present=r is not None,decision_started=a['decision_started'],wall_timeout=a['wall_timeout'],wall_seconds=a['wall_seconds'],attempt=a['attempt'],prefix_validation=jr['prefix_validation']if reused else a.get('prefix_validation'),original_prefix_validation=a.get('prefix_validation'),reused_equivalent_after_prefix_repair=reused,original_freeze_sha256=a['freeze_sha256'],mechanics=mechanics(trace)if(trace/'policy/plans.jsonl').exists()else None,source=str(p),source_sha256=sha(p),official_route_completed_km=float(r['meta']['route_length']*rc/100000)if r else None))
    done=[r for r in rows if r['status']!='PENDING'];complete=len(done)==24;arms={}
    for r in done:r['timeout_or_stall_derived']=timeout_or_stall(r)
    for arm in f['methods']:
        rr=[r for r in rows if r['method']==arm];dd=[r for r in rr if r['status']!='PENDING'];full=len(dd)==6
        arms[arm]=dict(expected=6,completed=len(dd),DS_mean_all6=float(np.mean([r['DS']for r in dd]))if full else None,success_all6=sum(r['success']for r in dd)if full else None,events_all6={e:sum(r['events'][e]for r in dd)if full and all(r['events'][e]is not None for r in dd)else None for e in EVENTS},known_event_subtotals={e:sum(r['events'][e]for r in dd if r['events'][e]is not None)for e in EVENTS},wall_seconds_completed=sum(r['wall_seconds']for r in dd),status='COMPLETE'if full else'PENDING_NO_PARTIAL_HEADLINE')
    selection=dict(status='PENDING_ALL_24',r_star=None)
    if complete:
        cw={r['route']:r for r in done if r['method']=='Map-CW'};simple={r['route']:r for r in done if r['method']=='Map-Simple'};official=all(r['event_complete']for r in done);interface=all((r.get('prefix_validation')or{}).get('passed',False)for r in done);ordinary=['2403','14909']
        def total(rr,events):return sum(r['events'][e]for r in rr.values()for e in events)
        conditions=dict(all_official_events=official,all_actual_prefix_gates=interface)
        if official:
            conditions.update(vehicle_and_pedestrian_not_increased=total(simple,['collisions_vehicle','collisions_pedestrian'])<=total(cw,['collisions_vehicle','collisions_pedestrian']),rule_penalized_routes_not_increased=sum(any(r['events'][e]for e in RULE)for r in simple.values())<=sum(any(r['events'][e]for e in RULE)for r in cw.values()),ordinary_no_added_stall=all(not(simple[r]['timeout_or_stall_derived']and not cw[r]['timeout_or_stall_derived'])for r in ordinary),mean_DS_strict_gain=arms['Map-Simple']['DS_mean_all6']>arms['Map-CW']['DS_mean_all6']+1e-6)
        selection=dict(status='NEEDS_FROZEN_RULE_REVIEW_AND_MATCHED_DEVELOPMENT',numeric_simple_candidate=all(conditions.values())and official,conditions=conditions,r_star=None,remaining_required=['Review all fixed/added safety events and unexplained regressions','If Simple numerically eligible, complete necessary Base/Generic-Simple matches within shared12 budget','Do not select or launch confirmation from this script'])
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='DEVELOPMENT_EXPOSED6',freeze_sha256=sha(freeze),completed=len(done),expected=24,complete=complete,arms=arms,selection=selection,rows=rows,primary_directions=['Map-r* minus Generic-r*','Map-CW minus Map-Simple'],not_confirmation=True,counterfactual_oracle_or_regret='MISSING: no complete true counterfactual utility labels in CARLA',official_success_definition='Completed/Perfect and zero infractions except min_speed_infractions, inherited final R18/pinned merge_route_json.py',missing_events='UNKNOWN, never zero-imputed; known subtotals separate',aggregation='All24 scheduled cells,6 exposed routes per arm; no full headline before all6',source_sha256=sha(Path(__file__)))
    out=RUN/'analysis/development';out.mkdir(parents=True,exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(out/f'SNAPSHOT_{stamp}.json').write_text(json.dumps(result,indent=2)+'\n');(out/'LATEST.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=len(done),expected=24,complete=complete,selection=selection)))
if __name__=='__main__':main()
