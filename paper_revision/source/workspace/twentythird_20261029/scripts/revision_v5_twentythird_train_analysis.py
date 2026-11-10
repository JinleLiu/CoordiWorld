"""TRAIN localpairedutility and supply diagnostics; not completeEPDMS."""
from revision_v5_twentythird_common import *
from revision_v5_seventeenth_strict_aggregation import METRICS
from revision_v5_sixteenth_common import macro
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import extract_features
import numpy as np,pandas as pd,argparse
def utility(c,ec):return np.prod(c[:4])*(5*c[4]+5*c[5]+2*c[6]+2*c[7]+2*ec)/16
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='oof');args=ap.parse_args();phase=args.phase;z=np.load(B22/'supply55/dataset.npz');oldraw=np.load(B22/'raw_nc/labels.npz')['physical_NC'].copy()
 for r in json.load(open(B22/'supply55/input_freeze.json'))['records']:oldraw[r['index'],r['new_slot']]=json.load(open(B22/'supply55/scenes'/r['scene_id']/'native.json'))['calls'][str(r['new_slot'])][0]['raw']['no_at_fault_collisions']
 reg=json.load(open(BASE/'dynamic80'/phase/'result.json'));rows=[];mech=[];rms=[]
 for r in reg['rows']:
  sid=r['scene_id'];i=r['index'];pa=np.load(BASE/'probe80'/phase/sid/'inputs.npz');pp=pa['previous_native'][r['offset']:];n=len(pp);fp=extract_features(pp[None],np.arange(n)*.1);cbase=z['components'][i];ec=pa['old_EC'];old=r['old_choice'];base=dict(scene_id=sid,drive=r['drive'],index=i,trigger=r['trigger'],old_choice=old,old_U=utility(cbase[old],ec[old]),old_NC=cbase[old,0],old_rawNC=oldraw[i,old],old_DAC=cbase[old,1],old_P=cbase[old,4],old_TTC=cbase[old,5],old_HC=cbase[old,7],old_EC=ec[old])
  for tag in ['probe80','dynamic80']:
   d=BASE/tag/phase/sid;nr=d/'native.json'
   if not nr.exists():continue
   nr=json.load(open(nr));rr=json.load(open(d/'record.json'));inp=np.load(d/'inputs.npz')
   if not nr['complete']:continue
   for k,name in enumerate(rr['variants']):
    c=np.array([nr['scores'][name][x]for x in METRICS[:8]]);ee=float(inp['new_EC'][k]);raw=nr['calls'][name][0]['raw'];row=base|dict(variant=name,U=utility(c,ee),EC=ee,NC=c[0],DAC=c[1],DDC=c[2],TLC=c[3],P=c[4],TTC=c[5],LK=c[6],HC=c[7],rawNC=raw[METRICS[0]],rawTTC=raw[METRICS[5]],delta_U=utility(c,ee)-base['old_U']);mech.append(row)
    ff=extract_features(inp['new_native'][k:k+1,:n],np.arange(n)*.1);rms.append(dict(scene_id=sid,variant=name,trigger=r['trigger'],EC=ee,**{x:float(np.sqrt(np.mean((ff[x]-fp[x])**2)))for x in ff}))
  fd=BASE/'train_features'/phase/sid
  if not(fd/'record.json').exists():continue
  rr=json.load(open(fd/'record.json'));f=np.load(fd/'features.npz');cs=cbase.copy();raws=oldraw[i].copy()
  if rr['admitted']:
   nr=json.load(open(BASE/'dynamic80'/phase/sid/'native.json'));comp=np.array([nr['scores']['dynamic2'][x]for x in METRICS[:8]]);cs=np.concatenate([cs,comp[None]]);raws=np.r_[raws,nr['calls']['dynamic2'][0]['raw'][METRICS[0]]]
  for policy in ['PNC','IC0','Geometry']:
   for suffix,ch in [('old',rr['old_choices'][policy]),('supply',rr['choices'][policy])]:
    c=cs[ch];ee=float(f['EC'][ch]);rows.append(dict(scene_id=sid,drive=r['drive'],policy=policy+'_'+suffix,choice=ch,new_slot=ch==12,U=utility(c,ee),EC=ee,rawNC=raws[ch],**dict(zip(['NC','DAC','DDC','TLC','P','TTC','LK','HC'],c))))
 out=BASE/'analysis'/phase;out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);md=pd.DataFrame(mech);df.to_csv(out/'selection.csv',index=False);md.to_csv(out/'candidate_diagnostics.csv',index=False);pd.DataFrame(rms).to_csv(out/'EC_rms.csv',index=False);sm={};mechanism={}
 for m,g in df.groupby('policy'):
  sm[m]=dict(n=len(g),drives=g.drive.nunique(),U=100*macro(g.U.values,g.drive.values),P=macro(g.P.values,g.drive.values),new_selected=int(g.new_slot.sum()),**{k:int((g[k]<1).sum())for k in ['NC','DAC','DDC','TLC','TTC','HC','EC','rawNC']})
 for name,g in md.groupby('variant'):
  t=g[g.trigger];comp=t[t.EC>0];mechanism[name]=dict(n=len(g),trigger_n=len(t),EC_compatible=len(comp),compatible_rawNC_fail=int((comp.rawNC<1).sum()),compatible_filteredNC_fail=int((comp.NC<1).sum()),compatible_DAC_fail=int((comp.DAC<1).sum()),compatible_U_improved=int((comp.delta_U>1e-8).sum()),compatible_HC_fail=int((comp.HC<1).sum()),candidate_P_mean=float(comp.P.mean())if len(comp)else None)
 write(out/'summary.json',dict(created_utc=now(),selection=sm,mechanism=mechanism,scope='80pairedTRAIN current-stage with EC, driveequal; NOT complete officialEPDMS'));print('B23_TRAIN',json.dumps(dict(selection=sm,mechanism=mechanism)),flush=True)
if __name__=='__main__':main()
