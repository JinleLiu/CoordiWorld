"""Input and first-divergence review: no official outcome fields used."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
RUN=Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z');sys.path.insert(0,str(RUN/'code'))
from audit_initial_inputs import normalized,differences

def load(final):
 fp=Path(final);tr=fp.parent/'attempt1/trace';frames=json.loads((tr/'first_four_frames.json').read_text());plans=[json.loads(s)for s in(tr/'policy/plans.jsonl').read_text().splitlines()];traces=[json.loads(s)for s in(tr/'trace.jsonl').read_text().splitlines()]
 with np.load(tr/'policy/plan_00000.npz')as a:arrays={k:a[k][:12].copy()for k in ['candidates','native','valid','x674']}
 return dict(frames=frames,normalized=normalized(frames),plans=plans,traces=traces,arrays=arrays,trace=tr)
def audit(a,b):
 diff=differences(a['normalized'],b['normalized']);categories={}
 for d in diff:
  key=d['path'].split('/')[2];categories[key]=categories.get(key,0)+1
 pool={k:bool(a['arrays'][k].shape==b['arrays'][k].shape and np.array_equal(a['arrays'][k],b['arrays'][k]))for k in a['arrays']}
 # Historical add_candidate explicitly updates pool-Geometry-relative P16[8:10].
 # These columns may change as a legitimate consequence of supplied candidates.
 invariant=[i for i in range(674)if i not in [666,667]]
 invariant_equal=np.array_equal(a['arrays']['x674'][:,invariant],b['arrays']['x674'][:,invariant])
 changed_columns=np.flatnonzero(np.any(a['arrays']['x674']!=b['arrays']['x674'],axis=0)).tolist()
 first=next((dict(replan=x['replan'],timestamps_us=[x['timestamp_us'],y['timestamp_us']],choices=[x['choice'],y['choice']],source_sha=[x['source_sha'],y['source_sha']],chosen_candidate_sha=[x['chosen_candidate_sha'],y['chosen_candidate_sha']],chosen_execution_sha=[x['chosen_execution_sha'],y['chosen_execution_sha']],kind='selected_trajectory_divergence_not_equal_history_claim')for x,y in zip(a['plans'],b['plans'])if x['chosen_candidate_sha']!=y['chosen_candidate_sha'] or x['chosen_execution_sha']!=y['chosen_execution_sha']),None)
 if first is None:first={'kind':'no_index_divergence_in_common_saved_replans','common_replans':min(len(a['plans']),len(b['plans']))}
 bm={r['step']:r for r in b['traces']};controls=[];pre=[]
 for x in a['traces']:
  y=bm.get(x['step'])
  if y is None:continue
  delta=max(abs(x['control'][k]-y['control'][k])for k in ['steer','throttle','brake'])
  if delta>1e-5 and not controls:controls.append(dict(step=x['step'],sim_seconds=x['sim_seconds'],abs_delta=delta))
  if not x['policy_replans']and not y['policy_replans']:pre.append(delta)
 return dict(normalized_initial_equal=not diff,initial_difference_count=len(diff),initial_difference_categories=categories,first25_differences=diff[:25],original12_pool_exact=pool,intrinsic_features_exact=bool(invariant_equal),changed_x674_columns=changed_columns,expected_pool_context_columns=[666,667],context_source='revision_v5_twentythird_features.add_candidate updates P16[8:10] when geometry anchor changes',unexplained_pool_mismatch_under_identical_inputs=not diff and not (all(pool[k]for k in ['candidates','native','valid'])and invariant_equal),first_selected_trajectory_divergence=first,first_actual_control_divergence=controls[0]if controls else None,prepolicy_max_control_delta=max(pre,default=None),initial_sha256=[hashlib.sha256((x['trace']/'first_four_frames.json').read_bytes()).hexdigest()for x in [a,b]],no_history_splicing=True,identical_seed_is_not_identical_observation=True)
def main(phase):
 v=RUN/'core_contribution_v05';q=v/(phase+'_v01');f=json.loads((v/'protocol'/f'{phase.upper()}_FREEZE_v01.json').read_text());final=q/'COORDINATOR_FINAL.json'
 if not final.exists():return dict(status='NOT_COMPLETE',no_scores_read=True)
 jobs=[json.loads(p.read_text())for p in sorted((q/'jobs').glob('*.json'))];assert len(jobs)==len(f['schedule']);lookup={(j['job']['route'],j['job']['seed'],j['job']['method']):j for j in jobs};out=[]
 for route,seed in sorted({(j['route'],j['seed'])for j in f['schedule']}):
  loaded={m:load(lookup[(route,seed,m)]['final'])for m in f['methods']}
  for a,b in [('H-Map','H-Generic'),('H-Map','H-Base')]:out.append(dict(route=route,seed=seed,arms=[a,b],**audit(loaded[a],loaded[b])))
 result=dict(status='COMPLETE_INPUT_AUDIT',phase=phase,pairs=out,passed=all(not r['unexplained_pool_mismatch_under_identical_inputs']for r in out),initial_mismatch_pairs=sum(not r['normalized_initial_equal']for r in out),no_scores_read=True)
 (v/'runtime'/f'{phase.upper()}_INPUT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(main(sys.argv[1])))
