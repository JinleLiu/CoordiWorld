# coding: utf-8
"""Minimal B R14 current-input path; original numerical bodies are loaded intact.
No policy training imports, new candidate production or evaluator calls.
"""
import os,sys,ast,json,hashlib,copy,time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
STAGE=Path(__file__).resolve().parents[1];PEER=STAGE/'peer_B_R14';ROOT=STAGE
PROJECT=STAGE.parent
RUNTIME=PEER/'source/dependencies/twentyfourth_20261029/runtime_src'
paths=[str(RUNTIME),str(PROJECT/'third_party/navsim'),str(PROJECT/'third_party/nuplan-devkit')]
paths += sorted({str(p.parent)for p in (PEER/'source').rglob('*.py') if 'scripts' in p.parts or 'crossline_current_map' in p.parts or p.parent.name=='R14'})
sys.path[:0]=paths
os.environ['NUPLAN_MAPS_ROOT']=str(ROOT/'data/maps');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch,yaml
from coordiworld.data.openscene import make_inputs,frame_agents,ego_pose,load_log
from coordiworld.data.maps import CurrentMapProvider
from coordiworld.data.pdm_proposer import PDMNominalProposer
from coordiworld.model import ModelConfig
from coordiworld.revision_v5.body_heading import BodyHeadingKinematicResidual
from coordiworld.revision_v5.common_coverage_b13 import CommonCoverageBaseWorld,common_reference_world
from coordiworld.revision_v5.current_history_comfort import current_history_comfort
from coordiworld.revision_v5.component_known_b13 import from_legacy,partial_current_DAC,geometry_known_score
from coordiworld.revision_v5.world_score_features import candidate_features
from revision_v5_sixteenth_semantics import semantic_features,current_areas
from revision_v5_seventeenth_progress import Extractor,p_context
from revision_v5_sixteenth_common import pair_features,choose,sha
from current_map_export import export_current_map
from deployment_features import build_trace
from revision_v5_twentysecond_kinematics import fix_s64,recover_legacy
from revision_v5_twentysecond_contact_features import contact_features
from revision_v5_eighteenth_source_features import features as vf
from revision_v5_eighteenth_road import features as rf
BINDINGS=[]
def upstream(pattern,name,namespace):
 p=next((PEER/'source').rglob(pattern));s=p.read_text();tree=ast.parse(s);node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name==name)
 # Use the unchanged function or class method AST, with explicit dependency bindings.
 if isinstance(node,ast.ClassDef):node=next(x for x in node.body if isinstance(x,ast.FunctionDef)and x.name=='inputs')
 BINDINGS.append(dict(source=str(p.relative_to(PEER)),source_sha256=sha(p),symbol=name,ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),namespace)
 return namespace[node.name]
b=SimpleNamespace(factual_inputs=upstream('*train_factual_metric_aligned.py','factual_inputs',dict(torch=torch,replace=replace)),execution_rules=upstream('*collect_paired_teachers_v2.py','execution_rules',dict(np=np,torch=torch)),candidate_features=candidate_features,export_current_map=export_current_map,build_trace=build_trace,static_features=upstream('*eleventh_map_augment.py','static_features',dict(np=np)))
class Pipeline:
 def __init__(self):
  from threadpoolctl import threadpool_limits
  self.thread_limit=threadpool_limits(limits=1)
  from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
  install_corrected_builder();self.config=yaml.safe_load(open(PEER/'config/group_cache_v2.yaml'))
  self.provider=CurrentMapProvider(ROOT/'data/maps');self.proposer=PDMNominalProposer(ROOT/'data/maps',self.config['candidates'])
  import revision_v5_seventeenth_progress as progress
  progress.ROOT=ROOT;self.progress=Extractor()
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  cp=PEER/'models/world/unqualified_model.pt';ck=torch.load(cp,map_location='cpu',weights_only=True)
  model=BodyHeadingKinematicResidual(ModelConfig(**ck['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state_dict'])
  self.world=CommonCoverageBaseWorld(ROOT/'data/maps',model);self.checkpoint_sha256=sha(cp)
Pipeline.inputs=upstream('*seventeenth_live.py','Pipeline',globals())
def load_history(row):
 p=Path(row['matches'][0]);assert sha(p)==row['source']['sha256']
 if row['source'].get('current_index')is not None:
  frames=load_log(p);i=row['source']['current_index'];h=frames[i-3:i+1]
 else:
  from navsim.common.dataclasses import Scene,SensorConfig
  from revision_v5_seventeenth_raw_history import raw_history_from_scene
  sc=Scene.load_from_disk(p,None,SensorConfig.build_no_sensors());h=raw_history_from_scene(sc)
 assert len(h)==4 and str(h[-1]['token'])==row['scene_id'];return h

def scene_for(pipe,h,candidates=None,valid=None):
 cfg=copy.deepcopy(pipe.config);cur=frame_agents(h[-1],np.asarray(ego_pose(h[-1])));cfg['data']['max_agents']=max(1,sum(np.linalg.norm(s[:2])<=cfg['data']['roi_m']for s,_ in cur.values()))
 bundle=pipe.proposer(h)if candidates is None else(candidates,valid,{'frozen_B_R14_bank':True})
 return make_inputs(h,cfg,pipe.provider(h,cfg['data']['max_map_tokens']),bundle)

def feature_view(out):
 ss,_=fix_s64(out['assets'],out['current'],out['S64'],out['S64_trace']);x=out['x'][0].copy();x[:,594:658]=ss;cf,trace=contact_features(out['assets'],out['current'],out['areas'])
 return dict(candidates=out['candidates'],valid=out['valid'],geo=np.array(out['Geometry_HC']),features198=out['features198'],x674=x,old_x674=out['x'][0],V24=vf(out['assets'])[0],R8=rf(out['areas'],out['valid']),C32=cf),trace
