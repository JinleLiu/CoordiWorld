"""Fixed-score/mask/history pool-availability decomposition on exposed DEV.

No model fitting or policy search. Both counterfactuals share the full H3-bank
PNC qualification computed before restricting availability to the first12.
First frames retain PNC. Complete scoring uses each submitted endpoint and EC.
"""
from revision_v5_twentysixth_common import *
import argparse,copy,shutil
import numpy as np
from revision_v5_twentythird_qualification import choice
from revision_v5_twentythird_progress_readout import select
STUDY=BASE/'supply_development'
METHODS=['PNC_H3P','Supply_M12_P0off','Supply_H3_P0off']

def prepare(scope):
    src=B24/'validation'/scope;out=STUDY/'validation'/scope
    if(out/'selection_freeze.json').exists():return
    write(STUDY/'registration.json',dict(created_utc=now(),reason='extend verified TRAIN fixed-score supply decomposition to both exposed DEV groups',hypothesis='availability gain versus P0 substitution can differ under complete mapping aggregation',fixed='same full H3 bank eligibility, frozen PNC readout, reference and previous plan; only accessibility and stated P0 score term differ',no_fit=True,no_parameter_search=True,no_unexposed_data=True,empty_M12_rule='fail and separately report; no adaptive fallback',source_sha=sha(__file__)))
    cohort=json.load(open(B24/'cohorts'/f'{scope}.json'));cohort['role']='EXPOSED_DEVELOPMENT_R16_posthoc_fixed_supply';write(STUDY/'cohorts'/f'{scope}.json',cohort)
    rows=copy.deepcopy(json.load(open(src/'choices.json')));details=[]
    for r in rows:
        check_time();sid=r['scene_id'];d=out/'scenes'/sid;d.mkdir(parents=True,exist_ok=True)
        choices={m:r['choices']['PNC_H3P']for m in METHODS};pd=src/'policy_inputs/PNC_H3P'/sid
        if(pd/'features.npz').exists():
            a=dict(np.load(pd/'features.npz'));rec=json.load(open(pd/'receipt.json'));fm={int(k):int(v)for k,v in rec['feature_to_bank'].items()}
            _,eligible=choice(a);off=a['PNC_final_scores'];mask=eligible&(np.arange(len(eligible))<12)
            assert mask.any(),(scope,sid,'fixed eligible M12 empty')
            p0on=select(a);assert fm[p0on]==choices['PNC_H3P']
            m12=int(np.where(mask,off,-np.inf).argmax());h3=int(np.where(eligible,off,-np.inf).argmax())
            choices.update(Supply_M12_P0off=fm[m12],Supply_H3_P0off=fm[h3])
            pr=a['PNC_projected'];formula=(5*pr[:,1]+5*pr[:,2]+2*pr[:,3]+2*a['features198'][:,197]*pr[:,0]+2*pr[:,0]*a['EC'])/16
            error=float(abs(formula[a['valid']]-off[a['valid']]).max());assert error<1e-6
            np.savez_compressed(d/'supply_inputs.npz',valid=a['valid'],eligible=eligible,scores_off=off,projected=pr,EC=a['EC'],P0=a['x674'][:,662],P0_known=a['x674'][:,668]>0,HC=a['features198'][:,197],raw_nc=a['PNC_raw_physical_nc'],raw_dac=a['PNC_raw_dac_event'],feature_to_bank=np.array([fm[j]for j in range(len(eligible))]),geo=a['geo'])
            details.append(dict(scene_id=sid,M12=m12,H3=h3,P0on=p0on,formula_error=error,source_feature_sha=sha(pd/'features.npz'),history=rec))
        r['choices']=choices;r['candidates_to_evaluate']=sorted(set(choices.values())|{int(r['reference'])});r['source_folder']=str(d)
        shutil.copy2(src/'scenes'/sid/'inputs.npz',d/'inputs.npz');write(d/'selection.json',r)
    by={r['scene_id']:r for r in rows}
    for g in cohort['plan']['groups']:
        gd=out/'groups'/str(g['registered_group_index']);ids=[s for pair in [(g['initial_token'],g['followup_token']),*g['synthetic_initial_followup_pairs']]for s in pair]
        write(gd/'choices.json',[by[s]for s in ids]);write(gd/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(gd/'choices.json'),new_native_labels_read=False))
    write(out/'choices.json',rows);write(out/'details.json',details);write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),new_native_labels_read=False,source_freeze_sha=sha(src/'selection_freeze.json'),formula_replay_scenes=len(details),PNC_selection_reproduced=len(rows),registration_sha=sha(STUDY/'registration.json')))
    print('SUPPLY_FROZEN',scope,len(rows),{m:sum(r['choices'][m]!=r['choices']['PNC_H3P']for r in rows)for m in METHODS},flush=True)

def native(scope,shard,shards):
    import revision_v5_twentysixth_development as m
    m.BASE=STUDY;m.native(scope,shard,shards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','native']);p.add_argument('scope');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    prepare(a.scope)if a.mode=='prepare'else native(a.scope,a.shard,a.shards)
