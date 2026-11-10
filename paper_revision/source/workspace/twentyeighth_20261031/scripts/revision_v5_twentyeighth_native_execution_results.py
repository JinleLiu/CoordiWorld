"""Separate native-execution block and same-route/seed execution×supply contrasts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_results import main,summary,paired,csvout,write,EVENTS
IDS=['World4Scorer_route_tp','World4Scorer_EC','World4Scorer_EC_H3','World4Scorer_EC_Gspeed','World4Scorer_NativeEC','World4Scorer_NativeEC_H3','World4Scorer_NativeEC_Gspeed']
def readrows(p):
 if not p.exists():return []
 rows=list(csv.DictReader(open(p)))
 for r in rows:
  for k in ['seed','success','completed100','control_steps','model_control_steps','efficiency_segments','efficiency_excluded_segments']+EVENTS:r[k]=int(r[k])if r.get(k)not in ['',None]else None
  for k in ['DS','RC','penalty','wall_seconds','duration_game','duration_system','driving_efficiency_percent']:r[k]=float(r[k])if r.get(k)not in ['',None]else None
  for k in ['has_official_record','wall_timeout']:r[k]=r.get(k)=='True'
 return rows
def run(raw,out):
 out.mkdir(parents=True,exist_ok=True);main(raw/'native_execution_run',out/'native');main(raw,out/'original');rows=readrows(out/'native/per_route_seed_method.csv')+readrows(out/'original/per_route_seed_method.csv');rows=[r for r in rows if r['method']in IDS and r['seed']==17];by={m:[r for r in rows if r['method']==m]for m in IDS};keys=set.intersection(*[{(r['route'],r['seed'])for r in by[m]}for m in IDS]);matched={m:[r for r in by[m]if(r['route'],r['seed'])in keys]for m in IDS};csvout(out/'all7_route_seed.csv',[r for m in IDS for r in matched[m]])
 contrasts=[(IDS[4],IDS[1]),(IDS[5],IDS[2]),(IDS[6],IDS[3]),(IDS[5],IDS[4]),(IDS[5],IDS[6]),(IDS[2],IDS[1]),(IDS[2],IDS[3])];pairs={}
 for a,b in contrasts:
  s,rr=paired(matched[a],matched[b]);pairs[a+'--'+b]=s;csvout(out/(a+'--'+b+'.csv'),rr)
 idx={(r['method'],r['route'],r['seed']):r for r in rows};did=[]
 for route,seed in sorted(keys):
  v=lambda m:idx[(m,route,seed)]['DS'];did.append(dict(route=route,seed=seed,new_H3_minus_EC=v(IDS[5])-v(IDS[4]),old_H3_minus_EC=v(IDS[2])-v(IDS[1]),delta_H3_increment=(v(IDS[5])-v(IDS[4]))-(v(IDS[2])-v(IDS[1]))))
 csvout(out/'difference_in_differences.csv',did);d=np.array([r['delta_H3_increment']for r in did]);ci=None
 if len(d)>1:
  rng=np.random.default_rng(17);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
 result=dict(scope='Already exposed Dev10, seed17 only; shared7-method intersection, no independent-confirmation claim',route_seeds=[list(k)for k in sorted(keys)],results={m:summary(matched[m])for m in IDS},paired=pairs,difference_in_differences=dict(definition='(NativeEC_H3−NativeEC)−(oldEC_H3−oldEC), DS points',mean=float(d.mean())if len(d)else None,route_bootstrap95=ci,n_routes=len(d)),limitations='Same route/weather/traffic seed, but camera tensors are not guaranteed bitwise identical across live runs. The published approximate dynamics and same native deployed PID are distinct.4s future validity not established by0.5s prefix check.')
 write(out/'comparison_summary.json',result);print(json.dumps(dict(common=len(keys),results={m:v['DS_route_equal']for m,v in result['results'].items()})))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw,a.out)
