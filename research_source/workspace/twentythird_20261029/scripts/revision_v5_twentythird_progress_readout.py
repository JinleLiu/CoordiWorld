"""Bounded new-candidate causalP0 diagnostic; original jointheads frozen.
No old-candidate score changes. Prior global P0 substitution failed inR07;
this tests only newEC1/rawNC/DAC-qualified additions after changed supply.
"""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_qualification import choice
from revision_v5_twentythird_train_analysis import utility
from revision_v5_sixteenth_common import macro
from revision_v5_seventeenth_strict_aggregation import METRICS
import numpy as np,pandas as pd
def select(a):
 ch,el=choice(a);s=a['PNC_final_scores'].copy();pr=a['PNC_projected'];mask=(np.arange(len(el))>=12)&(a['x674'][:,668]>0);s[mask]+=5/16*(pr[mask,0]*a['x674'][mask,662]-pr[mask,1]);return int(np.where(el,s,-np.inf).argmax())
def main():
 out=BASE/'progress_readout';out.mkdir(exist_ok=True);record('newP0progress','HYB complete408 improvement doesnotaddlinearly; candidateordering maymissprogress','Only newknownP0 proposals replace learnedQP byqhat*P0 invalue; keepHYBCAeligibility andallotherterms; onefixednohyperparameter trial','TRAIN80 original/backup; priorR07globalproxyfailure remains; do notclaimjointmomentunbiasedness');z=np.load(B22/'supply55/dataset.npz');result={}
 for phase in ['oof','robust']:
  root=BASE/'hybrid_features'/phase;rows=[];errors=[]
  for r in json.load(open(root/'selection_freeze.json'))['rows']:
   a=np.load(root/r['scene_id']/'features.npz');comps=list(z['components'][r['index']]);names=['old']*12
   for k in range(12,len(a['valid'])):
    item=r['candidate_mapping'][str(k)];d=BASE/item['tag']/phase/r['scene_id'];nr=json.load(open(d/'native.json'));name=json.load(open(d/'record.json'))['variants'][item['variant']];c=np.array([nr['scores'][name][x]for x in METRICS[:8]]);comps.append(c);names.append(name);q=np.prod(c[:4]);errors.append(dict(scene_id=r['scene_id'],variant=name,trueQP=q*c[4],predQP=a['PNC_projected'][k,1],P0QP=a['PNC_projected'][k,0]*a['x674'][k,662],eligible=bool(choice(a)[1][k])))
   old=choice(a)[0];ch=select(a);c=np.array(comps[ch]);rows.append(dict(scene_id=r['scene_id'],drive=r['drive'],choice=ch,old_choice=old,variant=names[ch],U=utility(c,a['EC'][ch]),oldU=utility(np.array(comps[old]),a['EC'][old]),EC=a['EC'][ch],**dict(zip(['NC','DAC','DDC','TLC','P','TTC','LK','HC'],c))))
  df=pd.DataFrame(rows);df.to_csv(out/(phase+'_choices.csv'),index=False);er=pd.DataFrame(errors);er.to_csv(out/(phase+'_errors.csv'),index=False);result[phase]=dict(U=100*macro(df.U.values,df.drive.values),deltaU=100*macro((df.U-df.oldU).values,df.drive.values),P=macro(df.P.values,df.drive.values),events={k:int((df[k]<1).sum())for k in ['NC','DAC','DDC','TLC','TTC','HC','EC']},changes=df[df.choice!=df.old_choice].to_dict('records'),QPmae=float(abs(er.trueQP-er.predQP).mean()),P0QPmae=float(abs(er.trueQP-er.P0QP).mean()))
 write(out/'result.json',dict(created_utc=now(),source_sha=sha(__file__),result=result));print('B23_P0',result,flush=True)
if __name__=='__main__':main()
