"""Read-only review of complete analysis: independent points/counts, input cause and costs.
No simulator calls, no changes to frozen statistics or original CSV.
"""
from pathlib import Path
import csv,collections,datetime,hashlib,json,statistics
B=Path(__file__).resolve().parents[1];O=B/'analysis/confirmation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=read(O/'FINAL_v01.json');z=read(B/'protocol/CONFIRMATION_FREEZE_v01.json');rows=f['rows'];guards=f['input_guards'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert f['completed']==f['scheduled']==216 and f['attempts']==216 and len(rows)==216
assert sha(B/'protocol/CONFIRMATION_FREEZE_v01.json')==f['freeze_sha256']=='c82e2bfdc750850cda5e6633144c79d8117abe8ad9019b3776c432b49dcb9a40'
key=lambda r:(str(r['route']),r['seed'],r['method'])
assert set(map(key,rows))==set(map(key,z['schedule'])) and len(set(map(key,rows)))==216
assert all(r['official_record_present'] and r['returncode']==0 and not r['wall_timeout'] and r['interface_validation']['passed']for r in rows)
assert all(not a['parse_errors']for a in f['attempt_rows'])
assert all(g['passed']for g in guards) and len(guards)==72
assert all(g['audit']['original12_pool'][k]['exact_equal']for g in guards for k in ['candidates','native','common_physical_mask'])
assert all(g['audit']['original12_pool']['x674']['exact_equal']for g in guards if g['audit']['normalized_observations_exact_equal'])
internal=[r for r in rows if r['mechanics']is not None];assert len(internal)==144
assert all(r['prefix_validation']['passed']for r in internal)
cs=list(csv.DictReader((O/'RESULTS_v01.csv').open(encoding='utf-8-sig')));counts=dict(collections.Counter(r['record_type']for r in cs));assert counts==f['csv']['record_type_counts'];assert sha(O/'RESULTS_v01.csv')==f['csv']['sha256']
ep=[r for r in cs if r['record_type']=='EPISODE'];assert len(ep)==216
for r in ep:
 s=next(x for x in rows if x['index']==int(r['index']));assert float(r['DS'])==s['DS'] and float(r['RC'])==s['RC']
points={}
for direction,p in f['primary'].items():
 a,b=direction.split(' minus ');d=[]
 for rt in z['routes']:
  dif=[next(x['DS']for x in rows if key(x)==(rt['id'],sd,a))-next(x['DS']for x in rows if key(x)==(rt['id'],sd,b))for sd in z['seeds']]
  d.append(dict(route=rt['id'],stratum=rt['stratum'],mean=statistics.mean(dif)))
 mean=statistics.mean(statistics.mean(r['mean']for r in d if r['stratum']==s)for s in sorted({r['stratum']for r in d}));assert abs(mean-p['mean'])<1e-10
 points[direction]=dict(independent_point=mean,positive_routes=sum(r['mean']>1e-6 for r in d),negative_routes=sum(r['mean']< -1e-6 for r in d),zero_routes=sum(abs(r['mean'])<=1e-6 for r in d),CI95=p['CI95'],CI97_5=p['CI97_5'])
mechanics={};latency={};sample_counts={};event_route_counts={}
for method,arm in f['arms'].items():
 rr=[r for r in rows if r['method']==method];assert len(rr)==36
 assert abs(statistics.mean(r['DS']for r in rr)-arm['DS_route_equal'])<1e-9
 assert sum(r['success']for r in rr)==arm['success_integer']
 latency[method]={x:statistics.mean(r['run_step_seconds'][x]for r in rr)for x in ['mean','p50','p95']}
 latency[method]['scope']='Mean of 36 episode summaries (balanced3seeds/12routes), not pooled percentile or realtime guarantee'
 sample_counts[method]={k:sum(r['motion'][k]for r in rr)for k in ['trace_samples','valid_trace_samples','invalid_trace_samples','metric_10Hz_samples','valid_metric_samples','invalid_metric_samples','metric_samples_without_exact_time_speed','moving_metric_samples','valid_jerk_intervals','excluded_jerk_time_gap_intervals','moving_jerk_intervals']}
 event_route_counts[method]={e:dict(episodes_with_event=sum((r['events'][e]or 0)>0 for r in rr),routes_with_event=len({r['route']for r in rr if (r['events'][e]or 0)>0}))for e in rr[0]['events']}
 if rr[0]['mechanics']is None:continue
 mm=[r['mechanics']for r in rr];m={k:sum(r.get(k,0)for r in mm)for k in ['replans','trigger_replans','selected_supply_replans','selected_shared_or_existing_stop_replans','no_physical_feasible_replans','selected_prefix_CV_conflict_replans','native_surrogate_covered_lane_replan_pairs_before','native_surrogate_covered_lane_replan_pairs_after','native_surrogate_new_covered_lane_replan_pairs','CW_risk_known_replans']}
 m['supply']={k:sum(r['supply'].get(k,0)for r in mm)for k in ['attempted','admitted','rejected']};rej=collections.Counter()
 for x in mm:rej.update(x['supply_rejections'])
 m['supply_rejections']=dict(rej);m['replan_seconds_p50_episode_mean']=statistics.mean(r['replan_seconds_p50']for r in mm);m['replan_seconds_p95_episode_mean']=statistics.mean(r['replan_seconds_p95']for r in mm)
 m['routes_with_selected_supply']=len({r['route']for r in rr if r['mechanics']['selected_supply_replans']>0});m['episodes_with_selected_supply']=sum(r['mechanics']['selected_supply_replans']>0 for r in rr)
 assert m['supply']['attempted']==m['supply']['admitted']+m['supply']['rejected'];mechanics[method]=m
review=dict(utc=now,analysis_sha256=sha(O/'FINAL_v01.json'),CSV_sha256=sha(O/'RESULTS_v01.csv'),freeze_sha256=f['freeze_sha256'],passed=True,scheduled_unique_cells=216,official_records=216,actual_attempts=216,bootstrap_failures=f['prestart_failure_count'],predecision_retry_budget_verified=f['predecision_retry_budget_verified'],CSV_record_counts=counts,independently_recomputed_primary_points=points,internal_prefix_passed=144,native_contract_passed=72,input_guards_passed=72,original12_candidates_native_mask_exact_pairs=72,features_exact_pairs=sum(g['audit']['original12_pool']['x674']['exact_equal']for g in guards),input_exact_pairs=sum(g['audit']['normalized_observations_exact_equal']for g in guards),mechanics=mechanics,run_step_seconds_episode_summary_means=latency,motion_sample_counts=sample_counts,event_incidence=event_route_counts,total_slot_hours=sum(r['all_attempt_wall_seconds']for r in rows)/3600,new_scientific_episodes=0)
(O/'FINAL_REVIEW_v01.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
mismatch=[g for g in guards if g['input_difference_requires_cause_review']];assert len(mismatch)==7
for g in mismatch:
 a=g['audit'];assert a['first_policy_steps']==[30,30] and a['pre_policy_samples_compared']==30
 assert all(a[k]==0 for k in ['pre_policy_max_control_delta','pre_policy_max_position_delta_m','pre_policy_max_velocity_delta_mps'])
 assert set(a['input_difference_categories']).issubset({'anns','current_support'})
sourceindex=read(O/'source_review/SOURCE_REVIEW_INDEX.json')
review2=dict(utc=now,freeze_unchanged=True,input_pair_count=72,exact_pairs=65,mismatch_pairs=7,all_original12_candidate_native_mask_pairs_exact=True,features_mismatched_pairs=2,prepolicy_steps_0_to29_control_position_velocity_equal_in_all7=True,first_policy_step=30,verified_findings=['All discrepancies are current actor annotations and associated current_support counts; no ego/map/controller field difference in normalized first4 observations.','Frozen common carla_view reads current live world actors; shared observed-support filter has no arm/readout branch.','Frozen scenario_manager launches independent wall-clock build_scenarios/spawn_parked_vehicles loop while synchronous scenario tick loop runs.','Four pairs have actor-count differences, three pairs only actor numeric differences. Identical initial inputs always give exact original12 pool/native/mask/features.','All144 internal actual-prefix gates and all72 native control/identity gates pass; Simple no-head contract was validated before frozen science.'],interpretation='Source and observations support asynchronous scenario spawn/timing and simulator numeric nondeterminism as explanations. Exact ordering of asynchronous calls was not instrumented; the precise cause of every difference remains unproven. This is a disclosed initial-condition limitation, not evidence of an arm-dependent input adapter or an unresolved known interface/control implementation error. Same seed does not mean identical input.',decision='Keep fixed all216 denominator and both comparisons. Report prespecified exact-first4 aligned sensitivity separately, excluding complete routes with any mismatched seed; do not retroactively change freeze, exclude failures or run more episodes. No new model or interface validity error identified by this review.',remaining_unknown=['Exact simulator/thread scheduling chain for each mismatch','Unobserved future actor evolution/counterfactual equality'],source_evidence=sourceindex,mismatch_records=mismatch,new_scientific_episodes=0)
(O/'INPUT_CAUSE_REVIEW_v01.json').write_text(json.dumps(review2,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:review[k]for k in ['passed','independently_recomputed_primary_points','mechanics','run_step_seconds_episode_summary_means','motion_sample_counts','total_slot_hours']},ensure_ascii=False,indent=2))
