"""Frozen historical bindings and light data adapters, without fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import ast,copy,csv,datetime,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('${CW_ARCHIVE_ROOT}')
R18=ROOT/'A_R18-research-20261031T1047';V03=ROOT/'A_R18-followup-v03-20261102T1447';V04=ROOT/'A_R18-turn-decision-v04-20261105T1832'
RUN=Path(__file__).resolve().parents[1]
C0='CoordiWorld-PNC_H3P';C1=C0+'_staticGuard_live_v01';S1='CoordiWorld-SimpleGuard-v05'
sys.path[:0]=[str(R18/'code'),str(V03/'code')]
import pnc_inference_v02 as pnc
from current_static_geometry_v01 import stopped_vehicle_guard
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from simple_rules import select as simple_select,guard_from_native
vehicle=get_pacifica_parameters()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');tmp.replace(p)
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify_implementation():
 cfg=read(RUN/'configs/IMPLEMENTATION_FREEZE_v01.json')
 for name,digest in cfg['files'].items():assert sha(name)==digest,(name,'frozen implementation/input changed')
 assert cfg['new_models']==0 and cfg['path']=='B'

def original_selector():
 """Extract the exact deployed C1 selector statements, including guard fallback."""
 source=R18/'code/repair_live_history_v01.py';cfg=read(RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')
 assert sha(source)==cfg['historical_C1_source_sha256']
 tree=ast.parse(source.read_text())
 # The original else branch contains the complete deployed selection statements.
 parent=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno==58)
 body=[copy.deepcopy(n) for n in parent.orelse if 89<=n.lineno<=99]
 assert [n.lineno for n in body]==[89,89,90,91,95,98,98,98,99]
 template=ast.parse("def select(a,h,ec,bankrec,unsafe):\n args=SimpleNamespace(policy='static_guard')\n return None\n").body[0]
 template.body=template.body[:1]+body+ast.parse("return before,choice,eligible,score").body
 ns={'np':np,'SimpleNamespace':SimpleNamespace};exec(compile(ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[])),str(source)+'#exact_selector_extract','exec'),ns)
 return ns['select']

def cohort():
 plan=read(R18/'queue_plans/static_guard_live_history_v01.json');out=[]
 for block in plan['blocks']:
  p=R18/'data'/f'cohort_block{block:03}.json';assert sha(p)==plan['cohort_pins'][p.name]
  out.extend(dict(row,block=block) for row in read(p)['records'])
 assert len(out)==2919 and sum(bool(x['scored']) for x in out)==1336
 return out

def c1_bank(row):
 t=row['token'];pred=R18/'predictions'/C1/t;rec=read(pred/'receipt.json');old=R18/'predictions'/C0/t
 if rec['status']=='PASS_INHERITED_FALLBACK':return None,rec
 if rec['live_reentry']:
  folder=R18/'analysis/static_guard_live_history_v01/live_cases'/t;geo=arrays(folder/'geometry.npz');bankrec=read(folder/'receipt.json')['bank_receipt']
 else:
  folder=old;gr=read(R18/'predictions/CoordiWorld-PNC_H3P_staticGuard_v01'/t/'receipt.json');geo=arrays(Path(gr['source_geometry']));bankrec=read(old/'receipt.json')
 a=arrays(folder/'candidate_features.npz');heads=arrays(folder/'head_outputs.npz');selector=arrays(pred/'selector.npz')
 assert len(a['valid'])==len(geo['execution_native']) and np.array_equal(a['valid'],geo['valid'])
 assert np.array_equal(a['candidates'][rec['choice']],np.load(pred/'trajectory.npy',allow_pickle=False))
 assert np.array_equal(selector['unsafe'],geo['unsafe'])
 ec=None if selector['own_EC'].size==0 else selector['own_EC']
 return dict(a=a,heads=heads,native=geo['execution_native'],unsafe=geo['unsafe'],ec_proxy=ec,bankrec=bankrec,
   bank_path=folder/'candidate_features.npz',bank_sha256=sha(folder/'candidate_features.npz'),selector=selector,geometry=geo),rec

def origin(bankrec,n):
 origins=['native_PDM']*min(12,n)
 # Brake replacement slot is not saved by the old producer. Do not invent it.
 if bankrec.get('brake_supply'):origins=['native_or_brake_slot_unrecorded']*min(12,n)
 origins+=['H3_'+x['variant'] for x in bankrec.get('variant_steps',[]) if x['admitted']]
 assert len(origins)==n
 return origins

def frozen_guard(native,raw_current,valid):
 origin=np.asarray(pnc.m.ego_pose(raw_current),float)
 cur={k:v for k,v in pnc.m.frame_agents(raw_current,origin).items() if np.linalg.norm(v[0][:2])<=100}
 current={'rear_axle_pose':origin,'states':np.asarray([v[0] for v in cur.values()],float).reshape(-1,7),'types':np.asarray([v[1] for v in cur.values()],int)}
 return guard_from_native(native,current,valid,stopped_vehicle_guard,vehicle)
