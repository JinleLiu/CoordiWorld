"""One TRAIN-motivated qualification repair, no threshold grid.
Only newly supplied EC-compatible candidate may use the DAC class boundary;
old-pool eligibility and raw-NC class boundary remain exactly frozen.
"""
from revision_v5_twentythird_common import *
import numpy as np,pandas as pd,argparse,copy,shutil
from revision_v5_twentythird_train_analysis import utility
from revision_v5_seventeenth_strict_aggregation import METRICS
from revision_v5_sixteenth_common import macro
def choice(a):
 el=a['PNC_eligible'].copy();extra=np.arange(len(el))>=12;el|=extra&a['valid']&(a['EC']>0)&(a['PNC_raw_dac_event']<=.5)&(a['PNC_raw_physical_nc']<=.5);return int(np.where(el,a['PNC_final_scores'],-np.inf).argmax()),el
def train():
 out=BASE/'qualification';out.mkdir(exist_ok=True);record('DACclass_newcandidate','New safe compatible candidate rejected for tiny DAC-risk difference from Geometry','Only appended EC1 candidate: predicted DACevent<=.5 and rawNC<=.5; preserve oldeligibility. Classification boundary, not calibrated probability guarantee','TRAIN80 original andfixedbackup ownPNC OOF; both must improve withnoNC/DAC/TTCincrease; no parameter grid');summary={}
 for phase in ['oof','robust']:
  z=np.load(B22/'supply55/dataset.npz');reg=json.load(open(BASE/'train_features'/phase/'selection_freeze.json'));records=[];base=pd.read_csv(BASE/'analysis'/phase/'selection.csv');base=base[base.policy=='PNC_supply'].set_index('scene_id')
  for r in reg['rows']:
   sid=r['scene_id'];a=np.load(BASE/'train_features'/phase/sid/'features.npz');ch,el=choice(a);old=base.loc[sid];c=z['components'][r['index'],ch]if ch<12 else np.array([json.load(open(BASE/'dynamic80'/phase/sid/'native.json'))['scores']['dynamic2'][k]for k in METRICS[:8]]);records.append(dict(scene_id=sid,drive=r['drive'],old_choice=int(old.choice),choice=ch,changed=ch!=old.choice,U=utility(c,a['EC'][ch]),oldU=old.U,EC=a['EC'][ch],oldEC=old.EC,**dict(zip(['NC','DAC','DDC','TLC','P','TTC','LK','HC'],c))))
  df=pd.DataFrame(records);df.to_csv(out/(phase+'_per_scene.csv'),index=False);summary[phase]=dict(U=100*macro(df.U.values,df.drive.values),deltaU=100*macro((df.U-df.oldU).values,df.drive.values),P=macro(df.P.values,df.drive.values),events={k:int((df[k]<1).sum())for k in ['NC','DAC','DDC','TLC','TTC','HC','EC']},changed=df[df.changed].to_dict('records'))
 write(out/'TRAIN_RESULT.json',dict(created_utc=now(),source_sha=sha(__file__),summary=summary));print('B23_QUAL_TRAIN',summary,flush=True)
def full(scope):
 src=BASE/'validation/continuation'/scope;out=BASE/'validation/continuation_qualification'/scope;out.mkdir(parents=True,exist_ok=True);assert not(out/'selection_freeze.json').exists();rows=copy.deepcopy(json.load(open(src/'choices.json')));contracts={r['current']:r for r in json.load(open(src/'history_contracts.json'))if r['policy']=='PNC'};changes=[]
 for r in rows:
  sid=r['scene_id'];ch=r['choices']['PNC_CONT']
  if sid in contracts:
   a=np.load(src/'policy_inputs/PNC'/sid/'features.npz');kk,el=choice(a);ch=contracts[sid]['feature_receipt']['bank_index']if kk>=12 else kk
  r['choices']['PNC_CA']=int(ch);r['candidates_to_evaluate']=sorted(set(r['choices'].values()));folder=out/'scenes'/sid;folder.mkdir(parents=True,exist_ok=True);write(folder/'selection.json',r)
  if ch!=r['choices']['PNC_CONT']:changes.append(sid)
  files=[src/'scenes'/sid/f'candidate_{c}_filtered.pkl'for c in r['candidates_to_evaluate']]
  if all(p.exists()for p in files):
   checks=[]
   for p in files:
    shutil.copy2(p,folder/p.name);p2=p.with_name(p.name.replace('_filtered.pkl','_physical.json'));shutil.copy2(p2,folder/p2.name);checks.append(dict(reused=True,source=str(p),sha256=sha(p),same_input_sha=r['input_sha']))
   write(folder/'native_receipt.json',dict(scene_id=sid,seconds=0,checks=checks,human_fixed=True))
 write(out/'choices.json',rows);write(out/'registration.json',dict(created_utc=now(),source_sha=sha(__file__),TRAIN_evidence_sha=sha(BASE/'qualification/TRAIN_RESULT.json'),parent_selection_sha=sha(src/'selection_freeze.json'),changed=changes,rule='appendedEC1, DAC<=.5 andrawNC<=.5; oldeligibility unchanged',exposure='alreadyexposedDEVELOPMENT'));write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),registration_sha=sha(out/'registration.json'),new_native_labels_read=False));print('B23_QUAL_FROZEN',scope,changes,flush=True)
 import revision_v5_twentythird_native as mod
 mod.BASE=BASE/'validation/continuation_qualification';mod.METHODS=list(rows[0]['choices']);mod.native(scope)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','full140','exposed408']);args=ap.parse_args();train()if args.mode=='train'else full(args.mode)
if __name__=='__main__':main()
