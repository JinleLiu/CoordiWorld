"""Join descriptive telemetry to exact method/cohort populations; no selection."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_results import csvout

def run(stats,diag):
 episodes={r['source_zip']:r for r in csv.DictReader(open(stats/'all_episodes.csv'))}
 key=lambda r:(r['run'],r['method'],r['route'],str(r['seed']))
 timing={key(r):r for r in csv.DictReader(open(diag/'all_timing.csv'))}
 actual={key(r):r for r in csv.DictReader(open(diag/'all_actual_metrics.csv'))}
 rows=list(csv.DictReader(open(stats/'method_cohort_summary.csv')))
 def route_mean(rr,column):
  by=defaultdict(list)
  for r in rr:
   if r.get(column)not in [None,'','None','NA']:by[r['route']].append(float(r[column]))
  return float(np.mean([np.mean(v)for v in by.values()]))if by else 'NA'
 for r in rows:
  ee=[episodes[p]for p in r['source_paths'].split('|')];tt=[timing[key(e)]for e in ee if key(e)in timing];aa=[actual[key(e)]for e in ee if key(e)in actual]
  r['telemetry_metric_episodes']=len(aa);r['telemetry_metric_coverage']=len(aa)/len(ee)
  for column in ['observed_horizontal_distance_km']:
   r['captured_'+column]=sum(float(v[column])for v in aa)if aa else 'NA'
  for column in ['upstream_literal_smoothness_fraction','stationary_fraction_v_le_0_1','long_accel_mps2_abs_p95','long_jerk_mps3_abs_p95','lat_accel_mps2_abs_p95']:
   r['route_equal_mean_episode_'+column]=route_mean(aa,column)
  for group in ['actual_moving_EC','actual_stationary_EC']:
   n=sum(int(x.get(group+'_n')or 0)for x in tt);z=sum(int(x.get(group+'_zero')or 0)for x in tt)
   label='moving_conditioned_surrogate_EC'if group=='actual_moving_EC'else'low_speed_conditioned_surrogate_EC'
   r[label+'_known_replans']=n;r[label+'_ones']=n-z;r[label+'_one_fraction']=(n-z)/n if n else 'NA'
  r['surrogate_EC_known_replans']=sum(int(x.get('EC_available')or 0)for x in tt)
  r['policy_replans_with_timing']=sum(int(x.get('replans')or 0)for x in tt)
  n=sum(int(x.get('planning_before_dump_s_n')or 0)for x in tt)
  r['planning_before_dump_call_weighted_mean_seconds']=sum(float(x.get('planning_before_dump_s_mean')or 0)*int(x.get('planning_before_dump_s_n')or 0)for x in tt)/n if n else 'NA'
  r['telemetry_note']='SI episode summaries averaged over seeds then routes; EC pooled across available replans and is surrogate execution, not observed CARLA comfort; captured distance may cover only a subset; no imputation.'
 csvout(stats/'method_cohort_summary_enriched.csv',rows)
 print(json.dumps(dict(rows=len(rows),actual_episode_records=len(actual),timing_episode_records=len(timing))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stats',type=Path);p.add_argument('diagnostics',type=Path);a=p.parse_args();run(a.stats,a.diagnostics)
