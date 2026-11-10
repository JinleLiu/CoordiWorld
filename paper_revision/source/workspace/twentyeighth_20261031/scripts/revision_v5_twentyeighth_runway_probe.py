"""Shared adapter: replay actual stalled terminal and ordinary parking inputs."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_policy import BridgePolicy
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
from revision_v5_twentyeighth_short_path import install as shortpath
from revision_v5_twentyeighth_reference_runway import install
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'map_supply/runway_probe';O.mkdir(parents=True,exist_ok=True);XML=B.parent/'twentyseventh_20261030/external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/data/drivetransformer_bench2drive_dev10.xml';polygons();prepared(False);shortpath();out=[]
ap=argparse.ArgumentParser();ap.add_argument('--route',default='25378');args=ap.parse_args()
for route,scope,method,last in [('25378','lowspeed_run/bench2drive','CoordiWorld_Full',True),('3514','bench2drive','CoordiWorld_BaseEC',False)]:
 if route!=args.route:continue
 d=B/scope/f'episodes/seed17/{route}/{method}/attempt1/trace';plans=[json.loads(x)for x in open(d/'policy/plans.jsonl')];row=plans[-1]if last else plans[0];hist=[json.loads(x)for x in open(d/'observed_history.jsonl')];hist=[h for h in hist if h['timestamp']<=row['timestamp_us']][-4:];goal=next(r for r in ET.parse(XML).getroot().findall('route')if r.get('id')==route).findall('./waypoints/position')[-1].attrib
 for h in hist:h['navigation_goal_math']=[float(goal['x']),-float(goal['y']),None]
 town=json.load(open(d/'agent_metadata.json'))['map_name'].split('/')[-1];p=BridgePolicy(json.load(open(B/f'bridge/maps/{town}.json')),'BaseEC',O/route,'controlled_v02');install(p);saved=np.load(d/'policy'/f"plan_{row['replan']:05d}.npz");p.prior=saved['previous_native']if row['history_timestamp_us']else None;p.prior_ts=row['history_timestamp_us'];ca,native,r=p.plan(hist);fresh=np.load(O/route/'plan_00000.npz');diff=float(np.max(abs(fresh['candidates'][:12]-saved['candidates'][:12])));ch=int(fresh['choice']);result=dict(route=route,source=str(d),source_replan=row['replan'],candidate_delta=diff,new_choice=ch,old_choice=int(saved['choice']),selected_4s_displacement=float(np.linalg.norm(native[-1,:2]-native[0,:2])),selected_pred_speed_1s=float(native[10,3]),rawNC=float(fresh['PNC_raw_physical_nc'][ch]),rawDAC=float(fresh['PNC_raw_dac_event'][ch]),EC=float(fresh['EC'][ch]),runway=r['reference_runway'])
 if route=='3514':assert diff==0
 out.append(result);print(json.dumps(result),flush=True)
(O/f'RESULT_{args.route}.json').write_text(json.dumps(dict(rows=out,scope='current saved input; original route goal unchanged; predicted change only, no real new episode'),indent=2)+'\n')
