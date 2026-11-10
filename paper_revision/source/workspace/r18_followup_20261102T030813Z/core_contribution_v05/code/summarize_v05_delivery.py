"""Descriptive final review of sealed estimates; no resampling, tuning, exclusions or simulation."""
from pathlib import Path
import json,hashlib,collections
from analyze_v05 import read,write_csv

def summarize(v):
 a=read(v/'analysis/FINAL_v01.json');assert a['completed']==72
 rows=a['rows'];idx={(r['route'],r['seed'],r['method']):r for r in rows};out=v/'analysis';pairs=[]
 for p in a['input_audit']['pairs']:
  ma,mb=p['arms'];ra=idx[p['route'],p['seed'],ma];rb=idx[p['route'],p['seed'],mb]
  pairs.append(dict(route=p['route'],seed=p['seed'],arm_a=ma,arm_b=mb,initial_equal=p['normalized_initial_equal'],DS_difference=ra['DS']-rb['DS'],RC_difference=ra['RC']-rb['RC'],difference_categories=p['initial_difference_categories'],first_selected_divergence=p['first_selected_trajectory_divergence'],interpretation='descriptive only; original full route-equal estimate unchanged'))
 strata=[]
 for mb in ['H-Generic','H-Base']:
  for eq in [True,False]:
   pp=[p for p in pairs if p['arm_b']==mb and p['initial_equal']==eq]
   strata.append(dict(arm_b=mb,initial_equal=eq,pairs=len(pp),routes=len({p['route']for p in pp}),pair_mean_DS_difference=sum(p['DS_difference']for p in pp)/len(pp)if pp else None,pair_mean_RC_difference=sum(p['RC_difference']for p in pp)/len(pp)if pp else None,not_primary_route_equal_estimator=True,not_a_second_significance_test=True))
 write_csv(out/'INITIAL_INPUT_DESCRIPTIVE_PAIRS.csv',pairs);write_csv(out/'INITIAL_INPUT_DESCRIPTIVE_STRATA.csv',strata)
 failures=[dict(phase=r['phase'],method=r['method'],route=r['route'],seed=r['seed'],status=r['status'],wall_timeout=r['wall_timeout'],returncode=r['returncode'],DS=r['DS'],RC=r['RC'],observed_seconds=r['motion']['observed_time_seconds'],source=r['source'],source_sha256=r['source_sha256'])for r in rows+a['bridge_rows']if r['status']not in ['Completed','Perfect']or r['returncode']!=0 or r['wall_timeout']]
 write_csv(out/'ALL_FAILURES.csv',failures)
 costs=[]
 for method,arm in a['arms'].items():
  rr=[r for r in rows if r['method']==method]
  # Aggregate stored means and counts; retain p95 as episode-p95 route-equal, never call it pooled p95.
  for metric,field in [('policy_planning_seconds','plan_seconds'),('base_pipeline_seconds','base_seconds'),('actual_run_step_seconds',None)]:
   vals=[r['mechanics'][field]if field else r['run_step_seconds']for r in rr];count=sum(x['n']for x in vals);total=sum(x['mean']*x['n']for x in vals if x['n'])
   costs.append(dict(method=method,metric=metric,count=count,total_seconds=total,per_call_weighted_mean_seconds=total/count if count else None,route_equal_mean_of_episode_p95_seconds=sum(x['p95']for x in vals)/len(vals),note='planning includes base pipeline; do not add overlapping timers; concurrent-slot wall time is not standalone latency'))
  costs.append(dict(method=method,metric='attempt_wall_seconds',count=len(rr),total_seconds=arm['all_attempt_wall_seconds'],mean_seconds=arm['all_attempt_wall_seconds']/len(rr)))
 write_csv(out/'ACTUAL_COST_TABLE.csv',costs)
 events=[]
 for mb in ['H-Generic','H-Base']:
  for r in rows:
   if r['method']!='H-Map':continue
   other=idx[r['route'],r['seed'],mb]
   for e in r['events']:
    x,y=r['events'][e],other['events'][e]
    if x is not None and y is not None and x!=y:
     events.append(dict(route=r['route'],seed=r['seed'],arm_a='H-Map',arm_b=mb,event=e,a_count=x,b_count=y,difference=x-y,a_events=r['event_details'][e],b_events=other['event_details'][e],a_source=r['source'],b_source=other['source'],interpretation='observed matched route-seed change, not proof of counterfactual causal event repair'))
 write_csv(out/'NEW_AND_REDUCED_EVENT_IDENTITIES.csv',events)
 review=dict(analysis_source_sha256=a['analysis_source_sha256'],fixed_analysis_sha256=hashlib.sha256((out/'FINAL_v01.json').read_bytes()).hexdigest(),confirmation_attempts=72,bridge_attempts=6,new_attempts_or_retries=0,initial_mismatch_pairs=[{k:p[k]for k in ['route','seed','arm_b','difference_categories','DS_difference']}for p in pairs if not p['initial_equal']],descriptive_strata=strata,confirmation_failed_status_by_arm={m:sum(r['status'].startswith('Failed')for r in rows if r['method']==m)for m in a['arms']},confirmation_interface_error_count=sum(bool(r['interface_errors'])for r in rows),confirmation_missing_official_count=sum(not r['official_record_present']for r in rows),confirmed_tick_runtime_cap=4000,tick_runtime_source='runtime/TICKRUNTIME_SOURCE_RECEIPT.json',failure_interpretation='Failed - TickRuntime is an existing platform termination after4000ticks, not process returncode failure; official blocked/timeout event zeros do not remove these failed episodes',primary_working_reference_met=False,primary_positive_route_count=a['comparisons']['PRIMARY']['positive_routes'],new_risk_identity='25865 seed42 H-Map: vehicle.ford.crown collision; Generic had no vehicle collision but failed at cap',adoption='Do not adopt stable unique Map-directed benefit; retain local limited signal and stop this supply optimization branch',no_reanalysis_of_old216=True,no_primary_exclusions=True)
 (out/'FINAL_REVIEW_v01.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n');return review
if __name__=='__main__':
 import sys;print(json.dumps(summarize(Path(sys.argv[1])),ensure_ascii=False))
