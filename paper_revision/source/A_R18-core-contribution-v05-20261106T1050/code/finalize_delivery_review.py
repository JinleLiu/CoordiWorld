"""Validate complete deliverables and correct only a saved chronological index.
No inference, native scoring, fitting or policy choice is executed here.
"""
import datetime,hashlib,json,shutil
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
def read(n):return json.loads((R/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
result=read('evidence/V05_FINAL_READOUT.json');events=pd.read_csv(R/'analysis/EVENT_TRANSITIONS.csv');drive=pd.read_csv(R/'analysis/PAIRED_DRIVE_COMPARISONS.csv');history=pd.read_csv(R/'analysis/S1_HISTORY_ACCOUNTING.csv');first=read('evidence/FIRST_SAVED_DIVERGENCES.json')
assert len(history)==2919 and history.token.nunique()==2919 and history.formal.sum()==1336
assert len(drive)==100 and drive.groupby('comparison').scenes.sum().eq(1336).all()
assert not events.duplicated(['comparison','token','metric']).any()
for comparison in result['all_comparisons']:
 for metric,m in comparison['metrics'].items():
  if 'added' not in m:continue
  x=events[(events.comparison==comparison['comparison'])&(events.metric==metric)]
  assert (x.transition=='added_by_left').sum()==m['added']
  assert (x.transition=='repaired_by_left').sum()==m['repaired']
oldline="if changed and row['drive'] not in first:first[row['drive']]=record"
newline="if changed and (row['drive'] not in first or record['timestamp_us']<first[row['drive']]['timestamp_us']):first[row['drive']]=record"
src=R/'code/analyze_v05.py';before_code=sha(src);before_first=sha(R/'evidence/FIRST_SAVED_DIVERGENCES.json');preserved=R/'failures/history_index_review';preserved.mkdir(parents=True,exist_ok=False)
shutil.copy2(src,preserved/'analyze_v05_before_index_fix.py');shutil.copy2(R/'evidence/FIRST_SAVED_DIVERGENCES.json',preserved/'FIRST_SAVED_DIVERGENCES_before.json')
assert src.read_text().count(oldline)==1
src.write_text(src.read_text().replace(oldline,newline));compile(src.read_text(),str(src),'exec')
corrections=[]
for d,x in list(first.items()):
 rows=history[(history.drive==d)&history.changed_from_C1];row=rows.loc[rows.timestamp_us.idxmin()]
 if x['timestamp_us']==row.timestamp_us:continue
 corrected=json.loads(row.to_json());corrected.pop('comparison',None)
 first[d]=corrected;corrections.append({'drive':d,'before_token':x['token'],'before_timestamp':x['timestamp_us'],'after_token':corrected['token'],'after_timestamp':corrected['timestamp_us']})
assert len(corrections)==2
write('evidence/FIRST_SAVED_DIVERGENCES.json',first)
for d,x in first.items():assert x['timestamp_us']==history[(history.drive==d)&history.changed_from_C1].timestamp_us.min()
write('evidence/HISTORY_INDEX_CORRECTION_v01.json',{'at':now,'classification':'delivery chronology index only','reason':'Original cohort preserves causal block execution order, which is not globally chronological across blocks of two drives; earliest lookup incorrectly used first encountered block. Corrected to minimum saved timestamp within drive.','original_own_generation':'unchanged; each input already checked its own original predecessor timestamp and preserved original block boundaries','corrections':corrections,'prior_index_sha256':before_first,'corrected_index_sha256':sha(R/'evidence/FIRST_SAVED_DIVERGENCES.json'),'analysis_code_before_sha256':before_code,'analysis_code_after_sha256':sha(src),'Score_or_events_changed':False,'new_scoring_calls':0,'rerun_own_history':False,'preserved_failure':str(preserved)})
w4=R.parent/'A_R18-followup-v03-20261102T1447/W4_PILOT_RECEIPT.json';dest=R/'evidence/prior_frozen/W4_PILOT_RECEIPT.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(w4,dest);assert sha(w4)==sha(dest)
write('evidence/prior_frozen/W4_SOURCE_RECEIPT.json',{'source':str(w4),'bytes':w4.stat().st_size,'sha256':sha(w4),'meaning':'execution obstruction, not a scientific negative; no rerun or download'})
verification={'at':now,'status':'PASS','official_formal_scenes':1336,'drives':20,'causal_inputs':2919,'event_rows':len(events),'event_count_identity_crosscheck':'all five comparisons match complete readout; no duplicate comparison/token/metric','first_saved_divergence':'timestamp minimum checked for all20 drives after preserved delivery-only correction','primary_full_effect_reviewed':True,'no_new_scoring':True,'readout_sha256':sha(R/'evidence/V05_FINAL_READOUT.json'),'first_divergences_sha256':sha(R/'evidence/FIRST_SAVED_DIVERGENCES.json')}
write('evidence/FINAL_READOUT_REVIEW_v01.json',verification)
s=read('STATE.json');s.update(updated_at=now,status='SCIENCE_COMPLETE_DELIVERY_PENDING',phase='FINAL_DELIVERY',mode='CONTINUOUS_DELIVERY',scientific_complete=True,full_v05_complete=False,active_scientific_queue='NONE_VERIFIED',latest_verified_queue=read('QUEUE_STATUS.json'),latest_queue_status='SCIENCE_COMPLETE_ANALYSIS_PENDING',latest_queue_phase='ANALYSIS_AND_DELIVERY',remaining_scientific_work='NONE: approved science and full readout complete; no additional experiment authorized',remaining_ETA='Final packaging, transfer and SHA checks only, approximately20-40min from '+now,final_readout='evidence/V05_FINAL_READOUT.json',final_readout_review='evidence/FINAL_READOUT_REVIEW_v01.json');write('STATE.json',s)
c=read('EXECUTION_CONTROL.json');c.update(updated_at=now,mode='CONTINUOUS_DELIVERY',current_mode='CONTINUOUS_DELIVERY',drivers={'scientific_queue':'COMPLETE; supervisor/children exited and locks released','delivery':'current chat; existing heartbeat remains until verified final sync then stop'},main_entry='this chat; do not submit new scientific jobs');write('EXECUTION_CONTROL.json',c)
(R/'NEXT_ACTIONS.md').write_text('# v05 final delivery checkpoint\n\nApproved science COMPLETE; final readout reviewed. Preserve original T0, T0_v05, frozen Simple A and pathB. No new fit, rule, strategy, W4, R19 or B change.\n\n1. Read evidence/V05_FINAL_READOUT.json and FINAL_READOUT_REVIEW_v01.json; official1336/20drive, 2919 inputs, all valid.\n2. Run code/package_v05.py --reviewed-complete-readout; same-version five files and post_export/V05_FINAL_EXPORT_RECEIPT.json.\n3. Download archive, verify remote byte/SHA, ZIP CRC and every MANIFEST member; attach four text/CSV files independently. Large assets stay originalAutoDL.\n4. Write and verify actual final document/state synchronization receipt. Formally stop this A heartbeat and mark current state COMPLETE_WAITING_CHIEF. No scientific continuation without next instruction.\n\nAll event identities, limitations and preserved chronology-index correction remain in package. Production source freezes unchanged. Old checkpoint/pause receipts are historical.\n')
print(json.dumps(verification));print(drive[drive.comparison=='STRATEGY_C1_minus_S1'].sort_values('delta_Score').to_json(orient='records'))
