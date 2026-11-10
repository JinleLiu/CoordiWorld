"""Keep actual runtime/controller replays separate from the frozen main cohort."""
import json,csv,argparse,hashlib
from pathlib import Path
from revision_v5_twentyeighth_results import EVENTS,paired,csvout
def collect(root):
 rows=[]
 for p in sorted((root/'bench2drive/episodes').glob('seed*/*/*/FINAL.json')):
  f=json.load(open(p));r=f.get('official_record')or{};sc=r.get('scores',{});ev=r.get('infractions',{});d=p.parent/f"attempt{f['attempt']}/trace";short=json.load(open(d/'short_path_stats.json'))if(d/'short_path_stats.json').exists()else{};low=json.load(open(d/'low_speed_stats.json'))if(d/'low_speed_stats.json').exists()else{}
  rows.append(dict(method=f['method'],route=f['route']['id'],seed=f['seed'],DS=float(sc.get('score_composed',0)),RC=float(sc.get('score_route',0)),penalty=sc.get('score_penalty'),status=r.get('status'),decision_started=f.get('decision_started'),success=int(r.get('status')in ['Completed','Perfect']and not any(v for k,v in ev.items()if k!='min_speed_infractions')),source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),config_sha=f['config_sha'],wall_seconds=f['wall_seconds'],short_path_calls=short.get('calls'),short_path_repairs=short.get('repaired'),released_low_speed_calls=low.get('released_calls'),**{e:len(ev.get(e,[]))if r else None for e in EVENTS}))
 return rows
def main(a):
 a.out.mkdir(parents=True,exist_ok=True);sources={'v02_main':collect(a.raw),'v03_short_path':collect(a.raw/'short_path_run'),'v04_lowspeed':collect(a.raw/'lowspeed_run')};allrows=[dict(scope=k,**r)for k,rows in sources.items()if k!='v02_main'for r in rows];csvout(a.out/'per_episode_repairs.csv',allrows);summaries={};gates=[]
 for new,old in [('v03_short_path','v02_main'),('v04_lowspeed','v03_short_path')]:
  for method in ['CoordiWorld_Full','CoordiWorld_BaseEC']:
   aa=[x for x in sources[new]if x['method']==method];bb=[x for x in sources[old]if x['method']==method];s,rows=paired(aa,bb);key=new+'--'+old+'--'+method;summaries[key]=s;csvout(a.out/(key+'.csv'),rows)
   if new=='v04_lowspeed':
    for r in rows:
     good=bool(r['delta_RC']>=-1e-9 and all(r['new_'+e]is not None and r['new_'+e]==0 for e in EVENTS if e!='min_speed_infractions'))
     gates.append(dict(method=method,route=r['route'],seed=r['seed'],safe_nonworse=good,positive=r['delta_DS']>1e-9 or r['delta_RC']>1e-9,delta_DS=r['delta_DS'],delta_RC=r['delta_RC']))
 eligible=all(x['safe_nonworse']for x in gates)and any(x['positive']for x in gates)if len(gates)==2 else None
 result=dict(scope='Separately registered actual replays. These do not overwrite original main outcomes. One route is not a generalization test; bootstrap omitted for one route.',comparisons=summaries,low_speed_gate=dict(expected_pairs=2,observed_pairs=len(gates),rows=gates,eligible_for_future_coverage=eligible,meaning='Predeclared descriptive pilot gate, not confirmation. Coverage remains unchanged until an explicit recorded freeze decision. min_speed segments are efficiency observations with unused DS penalty in pinned scorer.'))
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(episodes=len(allrows),low_speed_gate=result['low_speed_gate']),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
