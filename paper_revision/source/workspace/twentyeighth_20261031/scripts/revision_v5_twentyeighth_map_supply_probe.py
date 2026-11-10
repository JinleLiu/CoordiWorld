"""Same saved state/history: geometry pilot then exact frozen-head comparison."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_policy import BridgePolicy
from revision_v5_twentyeighth_map_supply_policy import install
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
from revision_v5_twentyeighth_short_path import install as shortpath
B=Path('<CW_WORKSPACE>/twentyeighth_20261031')
p=argparse.ArgumentParser();p.add_argument('--route',default='3514');p.add_argument('--indices',default='0,3');p.add_argument('--tag',default='probe_v02');a=p.parse_args();D=B/f'bench2drive/episodes/seed17/{a.route}/CoordiWorld_BaseEC/attempt1/trace';O=B/f'map_supply/{a.tag}/{a.route}';O.mkdir(parents=True,exist_ok=True)
h=[json.loads(s)for s in open(D/'observed_history.jsonl')];rows=[json.loads(s)for s in open(D/'policy/plans.jsonl')];town=json.load(open(D/'agent_metadata.json'))['map_name'].split('/')[-1];polygons();prepared(False);shortpath();policy=BridgePolicy(json.load(open(B/f'bridge/maps/{town}.json')),'BaseEC',O/'policy','controlled_v02');install(policy);result=[]
for i in map(int,a.indices.split(',')):
    row=rows[i];hist=[f for f in h if f['timestamp']<=row['timestamp_us']][-4:];assert len(hist)==4
    saved=np.load(D/'policy'/f'plan_{i:05d}.npz');policy.prior=None if i==0 else saved['previous_native'];policy.prior_ts=row['history_timestamp_us']
    (O/f'history_{i:05d}.json').write_text(json.dumps(hist)+'\n');ca,native,r=policy.plan(hist);new=np.load(O/'policy'/f'plan_{policy.n-1:05d}.npz');np.testing.assert_array_equal(new['candidates'][:12],saved['candidates'][:12]);r['original_choice']=int(saved['choice']);r['original12_exact']=True;result.append(r)
    print(json.dumps(dict(index=i,seconds=r['seconds'],choice=r['choice'],old_choice=r['original_choice'],supply=r['base_receipt']['map_supply'])),flush=True)
(O/'RESULT.json').write_text(json.dumps(dict(kind='saved current-input mechanism comparison; no counterfactual CARLA labels',rows=result),indent=2)+'\n')
