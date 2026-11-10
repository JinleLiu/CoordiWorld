"""Export compact initial/first-selected-divergence evidence from completed frozen runs."""
from pathlib import Path
import json,hashlib

def collect_windows(v,analysis,out):
 rows={(r['phase'],r['route'],r['seed'],r['method']):r for r in analysis['rows']+analysis['bridge_rows']};requests={key:{0:{'first_saved_replan_control'}}for key in rows}
 for phase,report in [('bridge',json.loads((v/'runtime/BRIDGE_INPUT_AUDIT.json').read_text())),('confirmation',analysis['input_audit'])]:
  for pair in report['pairs']:
   d=pair['first_selected_trajectory_divergence'];n=d.get('replan',0)
   for method in pair['arms']:requests[phase,pair['route'],pair['seed'],method].setdefault(n,set()).add('first_selected_trajectory_divergence'if 'replan'in d else'no_selected_trajectory_divergence_in_common_replans_control0')
 files={};index=[]
 for key,ns in requests.items():
  phase,route,seed,method=key;row=rows[key];tr=Path(row['source']).parent/'attempt1/trace';base=f'windows/{phase}/seed{seed}/{route}/{method}';plans={x['replan']:x for x in [json.loads(s)for s in(tr/'policy/plans.jsonl').read_text().splitlines()]};traces=[json.loads(s)for s in(tr/'trace.jsonl').read_text().splitlines()];starts={x['selected_replan']:i for i,x in enumerate(traces)if x.get('selected_replan')is not None};trace_sha=hashlib.sha256((tr/'trace.jsonl').read_bytes()).hexdigest();plans_sha=hashlib.sha256((tr/'policy/plans.jsonl').read_bytes()).hexdigest();observed_path=tr/'observed_history.jsonl';observed=[json.loads(s)for s in observed_path.read_text().splitlines()];observed_sha=hashlib.sha256(observed_path.read_bytes()).hexdigest();observed_at={int(r['timestamp']):i for i,r in enumerate(observed)};prefix_records={r['replan']:r for r in [json.loads(s)for s in(tr/'policy/v05_prefix.jsonl').read_text().splitlines()]}
  for rel in ['first_four_frames.json','policy/model_identity.json','v05_physics.json','agent_metadata.json']:
   p=tr/rel
   if p.exists():
    target=base+'/'+Path(rel).name;files[target]=p;index.append(dict(archive_path=target,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='initial observation/model binding'))
  for n,reasons in sorted(ns.items()):
   assert n in plans
   for name in [f'plan_{n:05d}.npz',f'v05_selected_prefix_{n:05d}.npz']:
    p=tr/'policy'/name;assert p.exists();target=base+'/'+name;files[target]=p;index.append(dict(archive_path=target,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason=sorted(reasons),replan=n))
   oi=observed_at[plans[n]['timestamp_us']];assert oi>=3;history=observed[oi-3:oi+1];assert hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest()==plans[n]['source_sha'];start=starts.get(n);assert start is not None;anchor=traces[start]['sim_seconds'];selected=[r for r in traces if anchor-.5-1e-6<=r['sim_seconds']<=anchor+.5+1e-6]
   target=base+f'/window_{n:05d}.json';p=out/'window_excerpts'/phase/f'seed{seed}'/route/method/f'window_{n:05d}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(phase=phase,route=route,seed=seed,method=method,replan=n,reasons=sorted(reasons),own_four_frame_history=history,history_source_sha256=plans[n]['source_sha'],observed_history_file=str(observed_path),observed_history_file_sha256=observed_sha,observed_history_line_numbers_1based=list(range(oi-2,oi+2)),prefix_record=prefix_records[n],plan_record=plans[n],actual_control_window=selected,plan_source=str(tr/'policy/plans.jsonl'),plan_source_sha256=plans_sha,trace_source=str(tr/'trace.jsonl'),trace_source_sha256=trace_sha,actual_trace_step_range=[selected[0]['step'],selected[-1]['step']],no_history_splicing=True,scope='one policy own recorded history; initial mismatch flags in pair audit'),ensure_ascii=False)+'\n');files[target]=p;index.append(dict(archive_path=target,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='derived exact excerpt with full-source SHA',replan=n))
 p=out/'V05_WINDOW_INDEX.json';p.write_text(json.dumps(index,indent=2)+'\n');return files
