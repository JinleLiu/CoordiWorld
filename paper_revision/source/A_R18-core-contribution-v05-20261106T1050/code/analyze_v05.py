"""Read only complete sealed v05 outputs; no tuning, fitting or evaluation."""
import collections,json
import numpy as np
import pandas as pd
import experiment_common as e

M={'NC':'no_at_fault_collisions','DAC':'drivable_area_compliance','DDC':'driving_direction_compliance','TLC':'traffic_light_compliance','EP':'ego_progress','TTC':'time_to_collision_within_bound','LK':'lane_keeping','HC':'history_comfort','EC':'two_frame_extended_comfort'}
EVENTS=['rawNC','NC','DAC','DDC','TLC','TTC','LK','HC','EC']

def interval(frame,value='delta_Score'):
 agg=frame.groupby('drive')[value].agg(['sum','count','mean']);rng=np.random.default_rng(170018);ix=rng.integers(len(agg),size=(10000,len(agg)));s=agg['sum'].to_numpy();n=agg['count'].to_numpy();boot=s[ix].sum(1)/n[ix].sum(1)
 return {'scenes':len(frame),'drives':len(agg),'scene_mean':float(frame[value].mean()),'drive_mean':float(agg['mean'].mean()),'drive_cluster_CI95':np.quantile(boot,[.025,.975]).tolist(),'positive_drives':int((agg['mean']>0).sum()),'negative_drives':int((agg['mean']<0).sum()),'zero_drives':int((agg['mean']==0).sum()),'bootstrap':{'resamples':10000,'seed':170018,'unit':'paired drive clusters','statistic':'scene-equal ratio sum(delta)/sum(n) across resampled drive clusters'}}

def compare(left,right,name,events):
 assert set(left.token)==set(right.token)
 assert left.token.is_unique and right.token.is_unique
 assert np.isfinite(left.Score.to_numpy()).all() and np.isfinite(right.Score.to_numpy()).all(), 'Do not silently shrink the official scene denominator'
 left=left.set_index('token').sort_index();right=right.set_index('token').loc[left.index];assert (left.drive==right.drive).all()
 out=pd.DataFrame({'token':left.index,'drive':left.drive.to_numpy(),'comparison':name,'left_Score':left.Score.to_numpy(),'right_Score':right.Score.to_numpy(),'delta_Score':left.Score.to_numpy()-right.Score.to_numpy()})
 summary={'comparison':name,**interval(out),'left_Score':float(left.Score.mean()),'right_Score':float(right.Score.mean()),'metrics':{}}
 for m in EVENTS+['EP']:
  l=pd.to_numeric(left[m],errors='coerce');r=pd.to_numeric(right[m],errors='coerce');known=l.notna()&r.notna()
  x={'paired_known':int(known.sum()),'left_known':int(l.notna().sum()),'right_known':int(r.notna().sum()),'left_mean_known':None if l.dropna().empty else float(l.mean()),'right_mean_known':None if r.dropna().empty else float(r.mean())}
  if m in EVENTS:
   added=known&(l<1)&(r>=1);repaired=known&(l>=1)&(r<1)
   x.update(left_events=int((l.dropna()<1).sum()),right_events=int((r.dropna()<1).sum()),added=int(added.sum()),repaired=int(repaired.sum()))
   for token in left.index[added|repaired]:
    record={'comparison':name,'token':token,'drive':left.loc[token,'drive'],'metric':m,'transition':'added_by_left' if added.loc[token] else 'repaired_by_left','left_value':float(l.loc[token]),'right_value':float(r.loc[token])}
    for side,table in [('left',left),('right',right)]:
     for k in ['candidate','candidate_origin','trajectory_id','previous_token','history_owner']:
      record[side+'_'+k]=table.loc[token,k] if k in table.columns else None
    events.append(record)
  summary['metrics'][m]=x
  out['left_'+m]=l.to_numpy();out['right_'+m]=r.to_numpy()
 return out,summary

def main():
 done=e.read(e.RUN/'evidence/S1_OWN_COMPLETE.json');assert done['status']=='PASS'
 conditional=e.read(e.RUN/'evidence/CONDITIONAL_COMPLETE.json');parity=e.read(e.RUN/'evidence/C1_CONDITIONAL_PARITY.json');assert conditional['status']==parity['status']=='PASS'
 condpath=e.RUN/'analysis/FOUR_ARM_CONDITIONAL.csv';assert e.sha(condpath)==conditional['result_sha256'];cond=pd.read_csv(condpath)
 ownpath=e.RUN/'analysis/S1_own/scored_tokens.csv';assert e.sha(ownpath)==done['scored_tokens_sha256']
 baseline=e.R18/'analysis/static_guard_live_history_v01'/e.C1/'scored_tokens.csv';assert e.sha(baseline)=='6c12cf12077ac0ed36a551a165781351d2cf3397b18edf7521c428719615188c'
 tables={};records={r['token']:r for r in e.cohort()};c1cond=cond[cond.arm=='C1'].set_index('token')
 for name,path in [('C1',baseline),('S1',ownpath)]:
  source=pd.read_csv(path);assert len(source)==1336 and source.token.nunique()==1336 and source.drive.nunique()==20
  assert source.evaluation_valid.eq(True).all(), 'Resolve and explicitly account for invalid evaluations before complete readout'
  t=source[['token','drive']].copy();t['Score']=100*source.headline_score;t['evaluation_valid']=source.evaluation_valid;t['history_owner']=name
  for key,col in M.items():t[key]=pd.to_numeric(source[col],errors='coerce')
  raw=[];origins=[];choices=[];ids=[];previous=[]
  for token in t.token:
   if name=='C1':
    x=c1cond.loc[token];raw.append(x.rawNC);origins.append(x.candidate_origin);choices.append(x.candidate);ids.append(x.trajectory_id);previous.append(x.previous_token)
   else:
    rec=e.read(e.RUN/'predictions'/e.S1/token/'receipt.json');lab=e.read(e.RUN/'results'/e.S1/token/'receipt.json');raw.append(lab['rawNC']);origins.append(rec['origin']);choices.append(rec['choice']);poses=np.load(e.RUN/'predictions'/e.S1/token/'trajectory.npy',allow_pickle=False);ids.append(e.hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest());previous.append(rec['history']['previous_token'])
  t['rawNC']=raw;t['candidate_origin']=origins;t['candidate']=choices;t['trajectory_id']=ids;t['previous_token']=previous;tables[name]=t
 events=[];scenes=[];summaries=[]
 primary,mainresult=compare(tables['C1'],tables['S1'],'STRATEGY_C1_minus_S1',events);scenes.append(primary);summaries.append(mainresult)
 for l,r in [('C1','S1'),('C0','S0'),('C1','C0'),('S1','S0')]:
  rows,summary=compare(cond[cond.arm==l].copy(),cond[cond.arm==r].copy(),'CONDITIONAL_'+l+'_minus_'+r,events);scenes.append(rows);summaries.append(summary)
 pd.concat(scenes,ignore_index=True).to_csv(e.RUN/'analysis/PAIRED_SCENE_COMPARISONS.csv',index=False)
 pd.concat([t.assign(method=n) for n,t in tables.items()],ignore_index=True).to_csv(e.RUN/'analysis/STRATEGY_PER_SCENE.csv',index=False)
 drive=pd.concat(scenes).groupby(['comparison','drive'],as_index=False).agg(scenes=('token','size'),left_Score=('left_Score','mean'),right_Score=('right_Score','mean'),delta_Score=('delta_Score','mean'));drive.to_csv(e.RUN/'analysis/PAIRED_DRIVE_COMPARISONS.csv',index=False)
 pd.DataFrame(events,columns=['comparison','token','drive','metric','transition','left_value','right_value']+[s+'_'+k for s in ['left','right'] for k in ['candidate','candidate_origin','trajectory_id','previous_token','history_owner']]).to_csv(e.RUN/'analysis/EVENT_TRANSITIONS.csv',index=False)
 # History divergence is read from the newly saved issued plans, never inferred by subtraction of counts.
 history=[];first={}
 for token,row in records.items():
  r=e.read(e.RUN/'predictions'/e.S1/token/'receipt.json');changed=r['changed_from_C1'];record={'token':token,'drive':row['drive'],'formal':bool(row['scored']),'timestamp_us':row['timestamp_us'],'changed_from_C1':changed,'changed_from_C0':r['changed_from_C0'],'live_reentry':r['live_reentry'],'guard_changed':r['guard_changed'],'candidate_origin':r['origin'],'previous_token':r['history']['previous_token'],'same_original_previous':r.get('same_original_previous'),'candidate_bank_sha256':r.get('bank_sha256'),'trajectory_sha256':r['trajectory_sha256']};history.append(record)
  if changed and (row['drive'] not in first or record['timestamp_us']<first[row['drive']]['timestamp_us']):first[row['drive']]=record
 pd.DataFrame(history).to_csv(e.RUN/'analysis/S1_HISTORY_ACCOUNTING.csv',index=False);e.write(e.RUN/'evidence/FIRST_SAVED_DIVERGENCES.json',first)
 mean=mainresult['scene_mean'];lo,hi=mainresult['drive_cluster_CI95'];risk={m:v['left_events']-v['right_events'] for m,v in mainresult['metrics'].items() if m in ['NC','DAC','DDC','TLC','TTC']}
 if mean<=0:decision='This comparison does not support a learned-evaluator advantage; no comparator replacement or further rule search. Non-significance is not equivalence.'
 elif mean>=.3 and lo>0 and not any(v>0 for v in risk.values()):decision='Supports a practically meaningful learned-evaluator increment within this exposed protocol; inspect all event identities and lineage limits before manuscript claims.'
 else:decision='Limited support only: positive mean with uncertainty, small practical magnitude, or risk cost; not a general advantage.'
 result={'at':e.now(),'status':'COMPLETE_READOUT','primary':mainresult,'all_comparisons':summaries,'decision':decision,'practical_reference_pp':.3,'reference_not_significance_threshold':True,'risk_event_net_C1_minus_S1':risk,'scope':'PathB exposed1336/20drive, shared learned proposer, evaluator replacement only; not independent confirmation','new_fits':0,'Simple_selected':'A_lexicographic','other_Simple_or_CW_choices_after_results':False,'historical_C1_reused':True,'negative_results_preserved':True,'history':{'all_changed_from_C1':sum(r['changed_from_C1'] for r in history),'formal_changed_from_C1':sum(r['changed_from_C1'] and r['formal'] for r in history),'live_reentries':sum(r['live_reentry'] for r in history),'first_divergence_drives':len(first)},'costs':{'conditional':conditional,'own_selection':{k:v for k,v in e.read(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json').items() if k!='records'},'own_evaluation':done},'result_files':{str(p.relative_to(e.RUN)):e.sha(p) for p in (e.RUN/'analysis').glob('*.csv')}}
 e.write(e.RUN/'evidence/V05_FINAL_READOUT.json',result);print(json.dumps({'primary':mainresult,'decision':decision,'history':result['history']},ensure_ascii=False))
if __name__=='__main__':main()
