"""Frozen conditional readouts on identical R14 PNC banks/prior and reference."""
from revision_v5_twentysixth_common import *
import argparse,copy,time,shutil
import numpy as np,xgboost as xgb,torch
from revision_v5_twentysixth_representation import representations
NEW=[a+'_'+p for p in ['M12','H3']for a in ['C','CV','Hybrid']]

def expanded_assets(base,a,current):
    asset={k:v.copy()for k,v in base.items()};M=len(a['valid']);asset['candidate_valid']=a['valid'].copy()
    if M>12:
        n=a['native'][12:];o=np.asarray(current['rear_axle_pose']);c,s=np.cos(o[2]),np.sin(o[2]);rot=np.array([[c,-s],[s,c]])
        loc=n.copy();loc[...,:2]=(n[...,:2]-o[:2])@rot;loc[...,2]=np.arctan2(np.sin(n[...,2]-o[2]),np.cos(n[...,2]-o[2]));loc[...,3:5]=n[...,3:5]@rot;loc[...,5:7]=n[...,5:7]@rot
        asset['execution_full_local']=np.concatenate([base['execution_full_local'],loc])
        asset['execution']=np.concatenate([base['execution'],loc[:,5::5,:3].astype(base['execution'].dtype)])
    assert len(asset['execution'])==M
    return asset

def prepare(scope):
    out=BASE/'validation'/scope;out.mkdir(parents=True,exist_ok=True);src=B24/'validation'/scope;old=B25/'validation'/scope
    if(out/'selection_freeze.json').exists():return
    cohort=json.load(open(B24/'cohorts'/f'{scope}.json'));cohort['role']='EXPOSED_DEVELOPMENT_R16_conditional_readout';write(BASE/'cohorts'/f'{scope}.json',cohort)
    heads={};bindings={}
    for name,arm in [('C','current'),('CV','cv'),('Hybrid','hybrid')]:
        p=BASE/'models/final/fit'/(arm+'.ubj');m=xgb.Booster();m.load_model(p);m.set_param({'nthread':1});heads[name]=m;bindings[name]=dict(path=str(p),sha256=sha(p))
    rows=copy.deepcopy(json.load(open(src/'choices.json')));start=time.monotonic();receipts=[];logs={'hashes':set()}
    for i,r in enumerate(rows):
        check_time();sid=r['scene_id'];d=out/'scenes'/sid;d.mkdir(parents=True,exist_ok=True);pd=src/'policy_inputs/PNC_H3P'/sid
        if(d/'selection.json').exists():r.update(json.load(open(d/'selection.json')));continue
        ch={m:int(r['choices']['PNC_H3P'])for m in NEW}
        if(pd/'receipt.json').exists():
            a=dict(np.load(pd/'features.npz'));rec=json.load(open(pd/'receipt.json'));fm={int(k):int(v)for k,v in rec['feature_to_bank'].items()}
            oldx=np.load(old/'scenes'/sid/'matched_predictions.npz');cur=json.load(open(src/'base'/sid/'current.json'));asset=expanded_assets(dict(np.load(src/'base'/sid/'world_assets.npz')),a,cur)
            x,rc,trace,cv=representations(asset,cur,oldx['current'],oldx['world'])
            if rc['old_Hybrid_feature_maxerror']>=1e-3:
                from revision_v5_twentysixth_execution_recovery import recover
                original_error=rc['old_Hybrid_feature_maxerror']
                asset,recovery=recover(dict(np.load(src/'base'/sid/'world_assets.npz')),a,cur,r,logs)
                x,rc,trace,cv=representations(asset,cur,oldx['current'],oldx['world'])
                rc['controller_recovery']=recovery|{'initial_roundtrip_feature_error':original_error}
                np.savez_compressed(d/'recovered_assets.npz',**asset);write(d/'current.json',cur)
            # Tiny coordinate round trips are recorded, never used to select a model.
            assert rc['old_Hybrid_feature_maxerror']<1e-3,(sid,rc['old_Hybrid_feature_maxerror'])
            pp={name:m.inplace_predict(x[{'C':'current','CV':'cv','Hybrid':'hybrid'}[name]])for name,m in heads.items()}
            np.testing.assert_array_equal(pp['C'],oldx['prediction_C'])
            for pool in ['M12','H3']:
                allow=a['valid']&((np.arange(len(a['valid']))<12)if pool=='M12'else True)
                for name in heads:ch[name+'_'+pool]=fm[int(np.where(allow,pp[name],-np.inf).argmax())]
            np.savez_compressed(d/'representations.npz',**x,**{n+'_prediction':p for n,p in pp.items()},**trace,valid=a['valid'],EC=a['EC'],candidate_indices=np.array([fm[j]for j in range(len(a['valid']))]))
            write(d/'representation_receipt.json',rc|dict(source_policy_input_sha=sha(pd/'features.npz'),old_matched_input_sha=sha(old/'scenes'/sid/'matched_predictions.npz'),reference=int(a['valid'].argmax()),history_contract=rec))
            receipts.append(rc)
        r['choices'].update(ch);r['candidates_to_evaluate']=sorted(set(r['choices'].values())|{int(r['reference'])})
        shutil.copy2(src/'scenes'/sid/'inputs.npz',d/'inputs.npz');assert sha(d/'inputs.npz')==r['input_sha'];r['source_folder']=str(d);write(d/'selection.json',r)
        if(i+1)%40==0:print('R16_DEV_PREPARE',scope,i+1,round(time.monotonic()-start,1),flush=True)
    by={r['scene_id']:r for r in rows}
    for g in cohort['plan']['groups']:
        gd=out/'groups'/str(g['registered_group_index']);ids=[s for p in [(g['initial_token'],g['followup_token']),*g['synthetic_initial_followup_pairs']]for s in p]
        write(gd/'choices.json',[by[s]for s in ids]);write(gd/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(gd/'choices.json'),models=bindings,new_native_labels_read=False))
    write(out/'choices.json',rows);write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),models=bindings,methods=list(rows[0]['choices']),new_native_labels_read=False,conditional_scope='same PNC initial selection, issued prior, bank, candidate/reference identities; each selected trajectory supplies its own endpoints and executed EC to full aggregation',seconds=time.monotonic()-start,model_freeze=sha(BASE/'models/final/fit_complete.json')))
    print('R16_DEV_FROZEN',scope,len(rows),round(time.monotonic()-start,1),flush=True)

def native(scope,shard=0,shards=1,aggregate_only=False):
    import revision_v5_twentyfourth_native as runner
    def index(s):
        z={}
        for root in [B24/'validation',B25/'validation',B25/'structured_development/validation']:
            src=root/s
            for r in json.load(open(src/'choices.json')):
                sid=r['scene_id'];d=src/'scenes'/sid
                for p in d.glob('candidate_*_physical.json'):
                    f=p.with_name(p.name.replace('_physical.json','_filtered.pkl'))
                    if f.exists():
                        j=json.load(open(p));z.setdefault(sid,{})[j['trajectory_sha']]=dict(filtered=f,physical=p,source_raw_sha=r['source']['sha256'])
        return z
    runner.BASE=BASE;runner.check_time=check_time;runner.METHODS=list(json.load(open(BASE/'validation'/scope/'choices.json'))[0]['choices']);runner.label_index=index
    runner.run(scope,shard,shards,aggregate_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','native']);p.add_argument('scope',choices=['full140','exposed408']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--aggregate-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    prepare(a.scope)if a.mode=='prepare'else native(a.scope,a.shard,a.shards,a.aggregate_only)
