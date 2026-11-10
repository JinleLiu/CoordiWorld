"""Diagnose a causal yield option under exactly the same original risk heads."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_policy import BridgePolicy,array_history
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
from revision_v5_twentyeighth_short_path import install as shortpath
from revision_v5_twentyeighth_reference_runway import install as runway
from revision_v5_twentythird_factory import ec_matrix
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'map_supply/gap_diagnostic';O.mkdir(parents=True,exist_ok=True);D=B/'bench2drive/episodes/seed17/3514/CoordiWorld_BaseEC/attempt1/trace';hh=json.load(open(D/'first_four_frames.json'));h=array_history(copy.deepcopy(hh));polygons();prepared(False);shortpath();p=BridgePolicy(json.load(open(B/'bridge/maps/Town13.json')),'BaseEC',O/'policy','controlled_v02');a,asset,native,current,r=p.base_inputs(p.pipe,h,hashlib.sha256(json.dumps(hh,sort_keys=True).encode()).hexdigest());old=p.model.predict(a)
# Actual stationary initial state, no invented reverse acceleration.
assert np.linalg.norm(h[-1]['ego_dynamic_state'][:2])<1e-6
ca=np.zeros((8,3),np.float32);a,asset,native,rr=p.add_candidate(p.pipe,h,'gap_probe',a,asset,current,ca);pred=p.model.predict(a);rows=[]
for i in list(np.flatnonzero(a['valid'])):
 rows.append(dict(i=int(i),choice=bool(i==pred['choice']),NC=float(pred['raw_physical_nc'][i]),DAC=float(pred['raw_dac_event'][i]),eligible=bool(pred['eligible'][i]),score=float(pred['final_scores'][i]),joint=pred['projected'][i].tolist(),C32=a['C32'][i].tolist(),P16=a['x674'][i,658:].tolist()))
np.savez_compressed(O/'pool.npz',**a,native=native,**{'PNC_'+k:v for k,v in pred.items()});(O/'RESULT.json').write_text(json.dumps(dict(original_choice=int(old['choice']),new_choice=int(pred['choice']),rows=rows),indent=2)+'\n');print(json.dumps(dict(original_choice=int(old['choice']),new_choice=int(pred['choice']),hold=rows[-1])),flush=True)
