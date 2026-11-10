"""Analyze the complete fixed12 matching block only. No partial scores or dispatch."""
from pathlib import Path
import sys,json,math,datetime,hashlib
RUN=Path(sys.argv[1])if len(sys.argv)>1 else Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z')
sys.path.insert(0,str(RUN/'versions/confirmation_v01'))
from confirmation_contract import internal_identity_validation
from analyze_development import mechanics,timeout_or_stall,EVENTS
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    q=RUN/'matched_development_v01';jobs=list((q/'jobs').glob('*.json'));out=RUN/'analysis/matched/FINAL_v01.json'
    if out.exists():print(json.dumps(dict(status='EXISTING_COMPLETE_ANALYSIS',path=str(out),sha256=sha(out))));return
    if len(jobs)!=12 or not(q/'COORDINATOR_FINAL.json').exists():print(json.dumps(dict(status='PENDING_COMPLETE_MATCHED_BLOCK',accounted=len(jobs),expected=12,no_outcomes_read=True)));return
    final=read(q/'COORDINATOR_FINAL.json');assert final['status']=='MATCHED_DEVELOPMENT_MATRIX_COMPLETE'and final['completed_jobs']==12
    assert all(read(p)['gate_passed']for p in jobs)and not read(q/'QUEUE_STATUS.json')['active']
    fp=RUN/'protocol/MATCHED_DEVELOPMENT_FREEZE_v01.json';f=read(fp);rows=[];identities=[]
    for i,j in enumerate(f['schedule']):
        jr=read(q/'jobs'/f'{i:02d}.json');p=Path(jr['final']);a=read(p);assert a['freeze_sha256']==sha(fp);method=f['methods'][j['method']]
        ad=p.parent/f"attempt{a['attempt']}";tr=ad/'trace';identity=internal_identity_validation(tr,method);identities.append(dict(job=j,**identity))
        r=a['official_record'];sc=(r or{}).get('scores',{});ev=(r or{}).get('infractions',{});status=(r or{}).get('status','MISSING_OFFICIAL_RECORD');ds=float(sc.get('score_composed',0));rc=float(sc.get('score_route',0));assert math.isfinite(ds)and math.isfinite(rc)
        if r:assert abs(ds-max(0.,rc*float(sc['score_penalty'])))<1e-3
        events={e:len(ev[e])if r and e in ev else None for e in EVENTS}
        row=dict(**j,actual_method_id=a['method'],actual_code_directory=a['code_directory'],actual_config_sha256=a['config_sha256'],actual_cohort=a['cohort'],status=status,DS=ds,RC=rc,success=int(bool(r)and status in ['Completed','Perfect']and not any(v for k,v in ev.items()if k!='min_speed_infractions')),events=events,event_complete=all(v is not None for v in events.values()),official_record_present=r is not None,decision_started=a['decision_started'],wall_timeout=a['wall_timeout'],wall_seconds=a['wall_seconds'],attempt=a['attempt'],prefix_validation=a['prefix_validation'],identity_validation=identity,mechanics=mechanics(tr)if(tr/'policy/plans.jsonl').exists()else None,source=str(p),source_sha256=sha(p),original_freeze_sha256=a['freeze_sha256'],official_route_completed_km=float(r['meta']['route_length']*rc/100000)if r else None)
        row['timeout_or_stall_derived']=timeout_or_stall(row);rows.append(row)
    initial=read(RUN/'analysis/development/INITIAL24_FINAL_v02.json');assert initial['complete']and initial['completed']==24
    arms={}
    for label in f['methods']:
        rr=[r for r in rows if r['method']==label];assert len(rr)==6
        arms[label]=dict(expected=6,completed=6,DS_mean_all6=sum(r['DS']for r in rr)/6,success_all6=sum(r['success']for r in rr),events_all6={e:sum(r['events'][e]for r in rr)if all(r['events'][e]is not None for r in rr)else None for e in EVENTS},wall_seconds_completed=sum(r['wall_seconds']for r in rr))
    allrows=initial['rows']+rows;by={(r['route'],r['method']):r for r in allrows};comparisons={}
    for aa,bb in [('Map-Simple','Generic-Simple'),('Map-CW','Map-Simple'),('Base-Simple','Base-CW'),('Generic-Simple','Generic-CW')]:
        paired=[dict(route=r['id'],difference=by[(r['id'],aa)]['DS']-by[(r['id'],bb)]['DS'])for r in read(RUN/'protocol/DEVELOPMENT_RULES.json')['routes']]
        comparisons[aa+' minus '+bb]=dict(pairs=paired,DS_difference_mean=sum(x['difference']for x in paired)/6,scope='exposed6 development only')
    pre=read(RUN/'protocol/SIMPLE_PRESELECTION_v01.json');conditions=dict(initial_preselection_all_conditions=all(pre['conditions'].values()),initial_no_unexplained_important_safety_regression=not pre['unexplained_important_safety_regression'],matched12_complete=True,matched_all_prefix_and_identity_gates=all(x['passed']for x in identities),all36_official_events_available=all(r['event_complete']for r in allrows))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='ADDITIONAL_MATCHED_DEVELOPMENT_EXPOSED6',complete=True,completed=12,expected=12,initial_completed=24,total_development=36,additional_budget_exhausted=True,freeze_sha256=sha(fp),initial_analysis_sha256=sha(RUN/'analysis/development/INITIAL24_FINAL_v02.json'),preselection_sha256=sha(RUN/'protocol/SIMPLE_PRESELECTION_v01.json'),arms=arms,rows=rows,comparisons=comparisons,identity_checks=identities,selection=dict(conditions=conditions,numeric_and_interface_simple_eligible=all(conditions.values()),requires_complete_input_audit_and_documented_rule_application=True,r_star_not_written_by_this_script=True),source_sha256=sha(Path(__file__)),not_confirmation=True,all_failures_retained=True)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(path=str(out),sha256=sha(out),complete=True,conditions=conditions)))
if __name__=='__main__':main()
