"""Share the causalP0 new-proposal value rule with the frozen IC0 control.
Its original eligibility remains unchanged: rank scores are not probabilities.
"""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_train_analysis import utility
from revision_v5_seventeenth_strict_aggregation import METRICS
from revision_v5_sixteenth_common import macro
import numpy as np,pandas as pd,argparse,shutil,copy
def select(a):
 p=a['IC0_projected'];s=a['IC0_final_scores'].copy();mask=(np.arange(len(s))>=12)&(a['x674'][:,668]>0);s[mask]+=5/16*(p[mask,0]*a['x674'][mask,662]-p[mask,1]);return int(np.where(a['IC0_eligible'],s,-np.inf).argmax())
def train(tag):
 out=BASE/'control_progress'/tag;out.mkdir(parents=True,exist_ok=True);z=np.load(B22/'supply55/dataset.npz');results={}
 for phase in ['oof','robust']:
  root=BASE/tag/phase;rows=[]
  for r in json.load(open(root/'selection_freeze.json'))['rows']:
   a=np.load(root/r['scene_id']/'features.npz');components=list(z['components'][r['index']])
   for k in range(12,len(a['valid'])):
    item=r['candidate_mapping'][str(k)];d=BASE/item['tag']/phase/r['scene_id'];nr=json.load(open(d/'native.json'));name=json.load(open(d/'record.json'))['variants'][item['variant']];components.append(np.array([nr['scores'][name][x]for x in METRICS[:8]]))
   ch=select(a);c=components[ch];old=r['choices']['IC0'];rows.append(dict(scene_id=r['scene_id'],drive=r['drive'],choice=ch,old_choice=old,U=utility(c,a['EC'][ch]),oldU=utility(components[old],a['EC'][old]),EC=a['EC'][ch],**dict(zip(['NC','DAC','DDC','TLC','P','TTC','LK','HC'],c))))
  df=pd.DataFrame(rows);df.to_csv(out/(phase+'.csv'),index=False);results[phase]=dict(U=100*macro(df.U.values,df.drive.values),deltaU=100*macro((df.U-df.oldU).values,df.drive.values),P=macro(df.P.values,df.drive.values),events={k:int((df[k]<1).sum())for k in ['NC','DAC','DDC','TLC','TTC','LK','HC','EC']},changes=df[df.choice!=df.old_choice].to_dict('records'))
 write(out/'result.json',dict(created_utc=now(),source_sha=sha(__file__),result=results));print('B23_CONTROL_P0_TRAIN',tag,results,flush=True)
def full(scope,spatial):
 tag='spatial_hybrid'if spatial else'hybrid';name='IC0_H3P'if spatial else'IC0_HP';parent='spatial_hybrid'if spatial else'progress_candidate';src=BASE/'validation'/parent/scope;inputroot=BASE/'validation'/tag/scope;out=BASE/'validation'/(tag+'_fair_progress')/scope;out.mkdir(parents=True,exist_ok=True);assert not(out/'selection_freeze.json').exists();rows=copy.deepcopy(json.load(open(src/'choices.json')));contracts={r['current']:r for r in json.load(open(inputroot/'history_contracts.json'))if r['policy']=='IC0'};oldname='IC0_H3'if spatial else'IC0_HYB';changes=[]
 for r in rows:
  sid=r['scene_id'];ch=r['choices'][oldname]
  if sid in contracts:
   a=np.load(inputroot/'policy_inputs/IC0'/sid/'features.npz');ch=contracts[sid]['feature_to_bank'][str(select(a))]
  r['choices'][name]=int(ch);r['candidates_to_evaluate']=sorted(set(r['choices'].values()));folder=out/'scenes'/sid;folder.mkdir(parents=True,exist_ok=True);write(folder/'selection.json',r)
  if ch!=r['choices'][oldname]:changes.append(sid)
  files=[src/'scenes'/sid/f'candidate_{c}_filtered.pkl'for c in r['candidates_to_evaluate']]
  if all(p.exists()for p in files):
   checks=[]
   for p in files:
    shutil.copy2(p,folder/p.name);p2=p.with_name(p.name.replace('_filtered.pkl','_physical.json'));shutil.copy2(p2,folder/p2.name);checks.append(dict(reused=True,source=str(p),sha256=sha(p),same_input_sha=r['input_sha']))
   write(folder/'native_receipt.json',dict(scene_id=sid,seconds=0,checks=checks,human_fixed=True))
 write(out/'choices.json',rows);write(out/'registration.json',dict(created_utc=now(),source_sha=sha(__file__),parent_sha=sha(src/'selection_freeze.json'),TRAIN_sha=sha(BASE/'control_progress'/(tag+'_features')/'result.json'),changed=changes,rule='same newcandidateP0value, IC0originalrank/DACeligibility unchanged; no DEVretuning',exposure='EXPOSED DEVELOPMENT'));write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),registration_sha=sha(out/'registration.json'),new_native_labels_read=False));print('B23_CONTROL_P0_FROZEN',scope,name,changes,flush=True)
 import revision_v5_twentythird_native as mod
 mod.BASE=BASE/'validation'/(tag+'_fair_progress');mod.METHODS=list(rows[0]['choices']);mod.native(scope)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','full140','exposed408']);ap.add_argument('--spatial',action='store_true');args=ap.parse_args()
 if args.mode=='train':train('spatial_hybrid_features'if args.spatial else'hybrid_features')
 else:full(args.mode,args.spatial)
