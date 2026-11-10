"""Portable CPU-only score replay. Does not run CARLA or alter a frozen package."""
import argparse,ast,hashlib,json,tempfile,sys,subprocess
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
def main(root,out):
 root=root.resolve();out=out.resolve();assert out!=root and root not in out.parents,'Verification output must be outside the frozen package';out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root/'code'))
 from revision_v5_twentyeighth_results import main as results
 from revision_v5_twentyeighth_ledger import main as ledger
 manifest=json.load(open(root/'MANIFEST.json'))
 listed={x['path']for x in manifest['artifacts']};actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'MANIFEST.json'}
 assert actual==listed,dict(unlisted=sorted(actual-listed),missing=sorted(listed-actual))
 for item in manifest['artifacts']:
  p=root/item['path'];data=p.read_bytes();assert len(data)==item['bytes']and hashlib.sha256(data).hexdigest()==item['sha256'],p
 by_path={x['path']:x for x in manifest['artifacts']}
 snapshot=json.load(open(root/'raw/SNAPSHOT_INDEX.json'))
 for source in snapshot['files']:
  item=by_path['raw/'+source['path']]
  assert item['bytes']==source['bytes']and item['sha256']==source['sha256'],source['path']
 expected=json.load(open(root/'results/summary.json'));results(root/'raw',out/'carla');ledger(root/'raw',out/'carla');got=json.load(open(out/'carla/summary.json'))
 def equal(a,b,path=''):
  if isinstance(a,dict):
   assert a.keys()==b.keys(),path
   for k in a:equal(a[k],b[k],path+'/'+k)
  elif isinstance(a,list):
   assert len(a)==len(b),path
   for i,(aa,bb)in enumerate(zip(a,b)):equal(aa,bb,path+'/'+str(i))
  elif isinstance(a,(int,float))and not isinstance(a,bool):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10,err_msg=path)
  else:assert a==b,(path,a,b)
 for key in ['common','paired','paired_common_W4_transfer4','common_by_seed','available_unmatched']:equal(expected[key],got[key],key)
 extras={}
 if(root/'results/expanded/studies.json').exists():
  from revision_v5_twentyeighth_all_results import run as expanded_results
  expanded_results(root/'raw',out/'expanded')
  equal(json.load(open(root/'results/expanded/studies.json')),json.load(open(out/'expanded/studies.json')),'expanded')
  if(root/'results/expanded/ATTRIBUTION_SENSITIVITY.json').exists():equal(json.load(open(root/'results/expanded/ATTRIBUTION_SENSITIVITY.json')),json.load(open(out/'expanded/ATTRIBUTION_SENSITIVITY.json')),'attribution_contribution_bookkeeping')
  extras['expanded']='All failed pilot cohorts, original frozen confirmation prefixes, matched effects and event deltas reproduced'
 if(root/'raw/research/w4_ec_scale_posthoc_v01/RESULT.json').exists():
  from revision_v5_twentyeighth_w4_ec_scale_posthoc import run as ec_scale
  ec_scale(root/'raw/research/w4_same_context_v03/scored_pools',out/'w4_ec_scale')
  equal(json.load(open(root/'raw/research/w4_ec_scale_posthoc_v01/RESULT.json')),json.load(open(out/'w4_ec_scale/RESULT.json')),'saved_W4_EC_scale')
  extras['w4_ec_scale']='Same-pool score arithmetic and53/47/6 selection counts reproduced; no counterfactual driving claim'
 if(root/'results/native_execution/comparison_summary.json').exists():
  from revision_v5_twentyeighth_native_execution_results import run as native_execution_main
  native_execution_main(root/'raw',out/'native_execution');equal(json.load(open(root/'results/native_execution/comparison_summary.json')),json.load(open(out/'native_execution/comparison_summary.json')),'native_execution');extras['native_execution']='same7-method block and execution-by-supply contrasts reproduced'
 if(root/'results/repairs/summary.json').exists():
  from revision_v5_twentyeighth_repairs import main as repair_main
  repair_main(argparse.Namespace(raw=root/'raw',out=out/'repairs'));equal(json.load(open(root/'results/repairs/summary.json')),json.load(open(out/'repairs/summary.json')),'repairs');extras['repairs']='all pilot contrasts and registered gate reproduced'
 if(root/'results/coverage/summary.json').exists():
  results(root/'raw/coverage_run',out/'coverage');expected_cov=json.load(open(root/'results/coverage/summary.json'));got_cov=json.load(open(out/'coverage/summary.json'))
  for key in ['common','paired','paired_common_W4_transfer4','common_by_seed','available_unmatched']:equal(expected_cov[key],got_cov[key],'coverage/'+key)
  extras['coverage']='separate v04 cohort reproduced'
 # NAVSIM: extract the two pure official aggregation functions from the exact
 # pinned runner. This avoids importing its simulator and model dependencies.
 import pandas as pd
 from typing import Dict,List,Tuple
 source=(root/'code/pinned_navsim_runner.py').read_text();tree=ast.parse(source);names={'calculate_weighted_average_score','calculate_individual_mapping_scores'}
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(selected)==2
 env=dict(np=np,pd=pd,Dict=Dict,List=List,Tuple=Tuple);exec(compile(ast.Module(body=selected,type_ignores=[]),'pinned_navsim_runner_aggregation','exec'),env)
 h=root/'history_control';tr=pd.read_csv(h/'train/per_scene.csv');trained=tr.groupby(['method','drive']).U.mean().groupby('method').mean()*100;te=json.load(open(h/'train/analysis.json'))['summary']
 for m,u in trained.items():np.testing.assert_allclose(u,te[m]['U16'],atol=1e-8,rtol=0)
 scopes={}
 for scope in ['full140','exposed408']:
  d=h/'validation'/scope;scene=pd.read_csv(d/'per_scene.csv');plan=json.load(open(d/'plan.json'));saved=pd.read_csv(d/'per_mapping.csv');rows={}
  mapping={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]for g in plan['groups']}
  for method,tab in scene.groupby('method'):
   scores=env['calculate_individual_mapping_scores'](tab[['token','weight','score']],mapping);means={}
   for stage,v in zip(['combined','stage_one','stage_two'],scores):
    truth=saved[(saved.method==method)&(saved.stage==stage)].score.mean();np.testing.assert_allclose(v['score'],truth,atol=1e-10,rtol=0);means[stage]=float(v['score']*100)
   rows[method]=means
  scopes[scope]=rows
 receipt=dict(verified_files=len(manifest['artifacts']),verified_original_snapshot_files=len(snapshot['files']),carla='all common, paired, per-seed and available summaries reproduced',additional=extras,TRAIN_U16=trained.to_dict(),NAVSIM=scopes,scope='Reaggregates saved real official outcomes and native candidate labels; does not rerun model inference, simulator execution, or all raw scoring.')
 (out/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();main(a.root,a.out or Path(tempfile.mkdtemp(prefix='coordiworld-r18-replay-')))
