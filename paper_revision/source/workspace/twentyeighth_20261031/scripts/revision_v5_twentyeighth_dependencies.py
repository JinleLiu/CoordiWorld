"""Capture actually imported private runtime sources plus precise model dependencies."""
import sys,json,hashlib,shutil,platform,datetime
from pathlib import Path
from revision_v5_twentyeighth_policy import BridgePolicy
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
ROOT=Path('<CW_WORKSPACE>');B=ROOT/'twentyeighth_20261031';O=B/'bridge/runtime_dependencies';O.mkdir(parents=True,exist_ok=True)
data=B/'bench2drive/episodes/seed17/25378/CoordiWorld_Full/attempt1/trace/first_four_frames.json';h=json.load(open(data));mapfile=B/'bridge/maps/Town03.json';p=BridgePolicy(json.load(open(mapfile)),'BaseEC',O/'fixture','controlled_v02');polygons();prepared(False);p.plan(h)
index={}
def add(path,kind,copy=True):
 path=Path(path).resolve()
 if not path.is_file()or not path.is_relative_to(ROOT):return
 rel=path.relative_to(ROOT);raw=path.read_bytes();entry=dict(original_path=str(path),root_relative_path=str(rel),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),kind=kind,included=copy)
 if copy:
  dest=O/'files'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);entry['package_path']=str(dest.relative_to(O))
 index[str(path)]=entry
for module in list(sys.modules.values()):
 path=getattr(module,'__file__',None)
 if not path:continue
 path=Path(path).resolve()
 if path.suffix=='.py'and path.is_relative_to(ROOT)and '/envs/'not in str(path)and '/external/'not in str(path):add(path,'imported_source')
for path in [data,mapfile,ROOT/'runtime_s1/configs/group_cache_v2.yaml',B/'scripts/cw_bridge_python.sh']:add(path,'fixture_or_configuration')
add(ROOT/'runs/dynamics_residual_batch_v1/kinematic_residual_s17/unqualified_model.pt','world_checkpoint')
for v in p.model.bindings.values():add(v['path'],'PNC_head')
import numpy,torch,xgboost,shapely
(O/'INDEX.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),versions=dict(numpy=numpy.__version__,torch=torch.__version__,xgboost=xgboost.__version__,shapely=shapely.__version__),source='One actual saved-context inference; imported source modules, exact frozen model files and canonical CARLA map fixture. Public pretrained weights/CARLA binary remain external.',files=list(index.values())),indent=2)+'\n')
print('captured',len(index),'files',sum(x['bytes']for x in index.values()),flush=True)
