"""Fixed saved states: shared support correction and terminal witness diagnostic."""
import json,hashlib
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_policy import BridgePolicy
from revision_v5_twentyeighth_map_supply import MapSupply
from revision_v5_twentyeighth_map_supply_policy import install as supply
from revision_v5_twentyeighth_map_timing_policy import install as timing
from revision_v5_twentyeighth_continuation_guard import install as guard
from revision_v5_twentyeighth_reference_runway import install as runway
from revision_v5_twentyeighth_observed_support import filtered_frame
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
from revision_v5_twentyeighth_short_path import install as shortpath
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'map_supply/continuation_probe_v02';O.mkdir(parents=True,exist_ok=True)
polygons();prepared(False);shortpath()
cases=[('mapsupply_timing_run','CoordiWorld_MapTiming_v01',0),('mapsupply_timing_run','CoordiWorld_MapTiming_v01',1),('','CoordiWorld_BaseEC',9),('mapsupply_timing_run','CoordiWorld_MapTiming_v01',69)]
policy=BridgePolicy(json.load(open(B/'bridge/maps/Town13.json')),'BaseEC',O/'policy','controlled_v02');fac=MapSupply(policy.map);supply(policy,fac.cmap);runway(policy);timing(policy,fac.cmap);guard(policy,fac.cmap);out=[]
for root,mid,index in cases:
    D=B/root/f'bench2drive/episodes/seed17/3514/{mid}/attempt1/trace';rows=[json.loads(l)for l in open(D/'policy/plans.jsonl')];r=rows[index];hh=[json.loads(l)for l in open(D/'observed_history.jsonl')];hist=[h for h in hh if h['timestamp']<=r['timestamp_us']][-4:];old=np.load(D/'policy'/f'plan_{index:05d}.npz')
    for mode in ['original_current_support','below_road_filtered']:
        h=hist if mode.startswith('original')else[filtered_frame(x,fac.cmap)for x in hist]
        policy.prior=None if index==0 else old['previous_native'];policy.prior_ts=r['history_timestamp_us'];n=policy.n;_,_,rec=policy.plan(h);a=np.load(O/'policy'/f'plan_{n:05d}.npz');g=rec['base_receipt']['terminal_viability'];item=dict(root=root,method=mid,original_index=index,mode=mode,original_choice=int(old['choice']),unguarded_choice=int(a.get('PNC_pre_guard_choice',a['choice'])),guarded_choice=int(a['choice']),guard_fallback=bool(a.get('PNC_guard_fallback',False)),viability=[dict(index=x['index'],viable=x['viable'])for x in g.get('checks',[])],removed=h[-1].get('current_support',{}).get('removed',[]),candidates=[dict(index=int(i),NC=float(a['PNC_raw_physical_nc'][i]),DAC=float(a['PNC_raw_dac_event'][i]),score=float(a['PNC_final_scores'][i]),eligible=bool(a['PNC_eligible'][i]))for i in np.flatnonzero(a['valid'])],policy_index=n)
        out.append(item);(O/f'history_{n:05d}.json').write_text(json.dumps(h)+'\n');print(json.dumps({k:v for k,v in item.items()if k!='candidates'}),flush=True)
(O/'RESULT.json').write_text(json.dumps(dict(kind='Same saved observations/history; no CARLA counterfactual outcomes',rows=out),indent=2)+'\n')
