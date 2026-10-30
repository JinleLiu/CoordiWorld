"""One registered matched pair; same target, folds, rows, capacity and budget.
The fit uses common H3 pools; M12 is an availability restriction of the same
head and bit-identical original-row features. Utility includes the common EC.
"""
from revision_v5_twentyfifth_common import *
import argparse,time
import numpy as np,pandas as pd,xgboost as xgb
from scipy.stats import spearmanr,kendalltau
from revision_v5_seventeenth_strict_aggregation import METRICS
from revision_v5_twentythird_qualification import choice as eligible_choice
from revision_v5_twentythird_progress_readout import select as pnc_select

SHORT=['NC','DAC','DDC','TLC','P','TTC','LK','HC']
PARAMS=dict(objective='reg:squarederror',tree_method='hist',max_leaves=15,max_depth=0,grow_policy='lossguide',eta=.05,reg_lambda=10,min_child_weight=5,seed=17,subsample=1,colsample_bytree=1,nthread=1,disable_default_eval_metric=True)
def utility(c,ec):return np.prod(c[:,:4],-1)*(5*c[:,4]+5*c[:,5]+2*c[:,6]+2*c[:,7]+2*ec)/16
def macro(x,drive):return float(np.mean([np.mean(x[drive==d])for d in sorted(set(drive))]))
def weights(scene,drive):
    sc,ct=np.unique(scene,return_counts=True);n_by_scene=dict(zip(sc,ct));drive_scenes={d:len(set(scene[drive==d]))for d in set(drive)}
    w=np.array([1/(drive_scenes[d]*n_by_scene[s])for s,d in zip(scene,drive)]);return w/w.mean()
def interval(v,seed=102819):
    rng=np.random.default_rng(seed);x=np.asarray(v,float);boot=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(point=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),positive_drives=int((x>1e-12).sum()),negative_drives=int((x< -1e-12).sum()),zero_drives=int((abs(x)<=1e-12).sum()),leave_one_drive_range=[float(((x.sum()-x)/(len(x)-1)).min()),float(((x.sum()-x)/(len(x)-1)).max())])

def dataset():
    path=BASE/'train/matched_dataset_v02.npz'
    if path.exists():return dict(np.load(path))
    reg=json.load(open(BASE/'train/manifest.json'));xx={'current':[],'world':[]};metadata=[];components=[];raw=[];ec=[];family=[];refchoices=[];offset=0;records=[]
    for ri,r in enumerate(reg['rows']):
        sid=r['scene_id'];d=BASE/'train/scenes'/sid;a=dict(np.load(d/'features.npz'));base=np.load(d/'base_features.npz');x=np.load(d/'matched_features_v02.npz');rec=json.load(open(d/'record.json'));complete=json.load(open(d/'labels/complete.json'));assert complete['complete']and complete['full_valid_pool'];ids=np.flatnonzero(a['valid']);cc=[];rr=[]
        for k in ids:
            p=json.load(open(d/'labels'/f'candidate_{k}_physical.json'));assert p['trajectory_sha']==array_sha(a['candidates'][k]);cc.append([p['filtered'][m]for m in METRICS[:8]]);rr.append([p['calls'][0]['raw'][m]for m in METRICS[:8]])
        for name in xx:xx[name].append(x[name][ids])
        components.extend(cc);raw.extend(rr);ec.extend(a['EC'][ids]);family.extend([rec['families'][k]for k in ids]);metadata.extend([dict(scene_id=sid,scene_index=ri,drive=r['drive'],fold=r['fold'],candidate=int(k),extra=bool(k>=12))for k in ids])
        pp=a['PNC_final_scores'].copy();mask=(np.arange(len(a['valid']))>=12)&(a['x674'][:,668]>0);pp[mask]+=5/16*(a['PNC_projected'][mask,0]*a['x674'][mask,662]-a['PNC_projected'][mask,1]);el=eligible_choice(a)[1]
        choices={}
        for pool in ['M12','H3']:
            allowed=a['valid']&(np.arange(len(a['valid']))<12)if pool=='M12'else a['valid'];src=base if pool=='M12'else a
            choices[pool]=dict(PNC=int(base['PNC_choice'])if pool=='M12'else pnc_select(a),IC0=int(base['IC0_choice'])if pool=='M12'else int(rec['choices']['IC0_H3P']),Geometry=int(src['Geometry_choice']))
        guard=el&~((np.arange(len(el))>=12)&((a['C32'][:,0]>0)|(a['C32'][:,1]>0)));gc=int(np.where(guard,pp,-np.inf).argmax())if guard.any()else int(base['PNC_choice']);choices['H3']['C32_new_guard']=gc
        source_masks={}
        for fam in ['history','current','spatial']:
            allow=el&((np.arange(len(el))<12)|(np.array(rec['families'])==fam));source_masks[fam]=int(np.where(allow,pp,-np.inf).argmax())if allow.any()else int(base['PNC_choice'])
        records.append(dict(scene_id=sid,drive=r['drive'],fold=r['fold'],row_start=offset,row_count=len(ids),candidate_ids=ids.tolist(),reference=int(a['valid'].argmax()),choices=choices,source_subset_choices=source_masks,source_mask_semantics='fixed full-bank predictions/context; alter availability only',trigger=rec['trigger'],family_count=len(a['valid'])-12));offset+=len(ids)
    meta=pd.DataFrame(metadata);meta.to_csv(BASE/'train/matched_rows_v02.csv',index=False);c=np.asarray(components,float);out={k:np.concatenate(v).astype(np.float32)for k,v in xx.items()};out.update(components=c,raw=np.asarray(raw,float),EC=np.asarray(ec,float),family=np.array(family),U=utility(c,np.asarray(ec)),**{k:meta[k].to_numpy()for k in meta})
    # Avoid object arrays in portable NPZ.
    for k,v in list(out.items()):
        if v.dtype.kind=='O':out[k]=v.astype(str)
    np.savez_compressed(path,**out);write(BASE/'train/matched_scene_records_v02.json',records);write(BASE/'train/DATASET_FREEZE_v02.json',dict(complete=True,created_utc=now(),manifest_sha=sha(BASE/'train/manifest.json'),dataset_sha=sha(path),scenes=len(records),drives=len(set(out['drive'])),candidate_rows=len(meta),extra_rows=int(out['extra'].sum()),full_valid_pool=True,schema_sha=sha(BASE/'train/FEATURE_SCHEMA_v02.json'),protocol=PIN,domain=DOMAIN,exposure='TRAIN mechanism; head OOF conditional on fixed per-scene OOF old-policy bank'))
    return out

def fit(final=False):
    z=dataset();out=BASE/'matched/v02'/('final'if final else'oof');out.mkdir(parents=True,exist_ok=True);assert xgb.__version__=='2.1.4';reg=json.load(open(BASE/'EXPERIMENT_REGISTRATION.json'));write(out/'registration.json',dict(created_utc=now(),source_sha=sha(__file__),data_sha=sha(BASE/'train/matched_dataset_v02.npz'),params=PARAMS,rounds=150,registration=reg['paired_head'],side_names=['current','world'],xgboost=xgb.__version__,feature_correction='All current-derived S64/C32/V24 blocks and old current/map features now shared; v01 is a mixed-representation preliminary control, not pure forecast attribution'))
    pred={k:np.full(len(z['U']),np.nan)for k in ['current','world']};receipts=[];t=time.monotonic()
    for fold in ([None]if final else range(3)):
        check_time();train=np.arange(len(z['U']))if fold is None else np.flatnonzero(z['fold']!=fold);hold=np.arange(len(z['U']))if fold is None else np.flatnonzero(z['fold']==fold);train=train[np.lexsort((z['candidate'][train],z['scene_id'][train]))];w=weights(z['scene_id'][train],z['drive'][train]);d=out/('fit'if fold is None else'fold'+str(fold));d.mkdir(exist_ok=True);np.savez_compressed(d/'layout.npz',train=train,hold=hold,weights=w);record={}
        for name in ['current','world']:
            m=xgb.train(PARAMS,xgb.DMatrix(z[name][train],label=z['U'][train],weight=w,nthread=1),num_boost_round=150,verbose_eval=False);m.save_model(d/(name+'.ubj'));write(d/(name+'_config.json'),json.loads(m.save_config()));pred[name][hold]=m.inplace_predict(z[name][hold]);fp=m.inplace_predict(z[name][train]);np.savez_compressed(d/(name+'_fit.npz'),indices=train,predictions=fp);tree=m.trees_to_dataframe();record[name]=dict(model_sha=sha(d/(name+'.ubj')),dimensions=m.num_features(),trees=m.num_boosted_rounds(),leaves=int((tree.Feature=='Leaf').sum()),fit_weighted_MAE=float(np.average(abs(fp-z['U'][train]),weights=w)),fit_weighted_MSE=float(np.average((fp-z['U'][train])**2,weights=w)))
        assert set(z['drive'][train]).isdisjoint(set(z['drive'][hold]))or final;write(d/'receipt.json',record);receipts.append(dict(fold=fold,heads=record));print('R15_MATCHED_FIT',fold,record,'sec',round(time.monotonic()-t,1),flush=True)
    np.savez_compressed(out/'predictions.npz',**pred);write(out/'fit_complete.json',dict(complete=True,created_utc=now(),seconds=time.monotonic()-t,receipts=receipts));analyze(z,pred,out)

def analyze(z,pred,out):
    recs=json.load(open(BASE/'train/matched_scene_records_v02.json'));rows=[];rankrows=[];source_rows=[]
    for r in recs:
        ii=np.arange(r['row_start'],r['row_start']+r['row_count']);lookup=dict(zip(z['candidate'][ii],ii))
        for pool in ['M12','H3']:
            ids=ii[z['candidate'][ii]<12]if pool=='M12'else ii;oracle=float(z['U'][ids].max());choices={m:lookup[k]for m,k in r['choices'][pool].items()};choices.update({m:int(ids[np.argmax(p[ids])])for m,p in pred.items()})
            for method,k in choices.items():
                c=z['components'][k];raw=z['raw'][k];rows.append(dict(pool=pool,method=method,scene_id=r['scene_id'],drive=r['drive'],fold=r['fold'],candidate=int(z['candidate'][k]),family=str(z['family'][k]),U=float(z['U'][k]),EC=float(z['EC'][k]),oracle_U=oracle,regret=oracle-float(z['U'][k]),rawNC=float(raw[0]),rawDAC=float(raw[1]),rawDDC=float(raw[2]),rawTLC=float(raw[3]),**dict(zip(SHORT,c.tolist()))))
            for method,p in pred.items():
                yy=z['U'][ids];ss=p[ids];diff=yy[:,None]-yy;pdiff=ss[:,None]-ss;mask=np.triu(diff!=0,1);pairs=int(mask.sum());correct=float(((diff[mask]*pdiff[mask]>0)+.5*(pdiff[mask]==0)).sum());eligible=len(ids)>1 and np.ptp(yy)>0 and np.ptp(ss)>0
                rankrows.append(dict(pool=pool,method=method,scene_id=r['scene_id'],drive=r['drive'],candidates=len(ids),pairs=pairs,correct=correct,MAE=float(abs(ss-yy).mean()),spearman=float(spearmanr(yy,ss).statistic)if eligible else None,kendall_tau_b=float(kendalltau(yy,ss).statistic)if eligible else None,rank_defined=eligible,prediction_range=float(np.ptp(ss)),truth_range=float(np.ptp(yy))))
        for fam,k in r['source_subset_choices'].items():
            i=lookup[k];source_rows.append(dict(scene_id=r['scene_id'],drive=r['drive'],available_family=fam,candidate=int(k),U=float(z['U'][i]),NC=float(z['components'][i,0]),DAC=float(z['components'][i,1]),P=float(z['components'][i,4]),EC=float(z['EC'][i])))
    df=pd.DataFrame(rows);df.to_csv(out/'per_scene.csv',index=False);rd=pd.DataFrame(rankrows);rd.to_csv(out/'ranking.csv',index=False);pd.DataFrame(source_rows).to_csv(out/'source_masks.csv',index=False);df.groupby(['pool','method','drive'])[['U','P','EC','regret']].mean().reset_index().to_csv(out/'per_drive.csv',index=False)
    summary={};comparisons={}
    for pool in ['M12','H3']:
        pp=df[df.pool==pool];summary[pool]={}
        for method,g in pp.groupby('method'):
            drive=g.drive.to_numpy();summary[pool][method]=dict(scenes=len(g),drives=len(set(drive)),U=100*macro(g.U.to_numpy(),drive),P=macro(g.P.to_numpy(),drive),oracle_regret=100*macro(g.regret.to_numpy(),drive),EC=macro(g.EC.to_numpy(),drive),events={m:int((g[m]<1).sum())for m in ['NC','DAC','DDC','TLC','TTC','HC','EC','rawNC']},new_selected=int((g.candidate>=12).sum()))
        for new,old in [('world','current'),('world','PNC'),('current','PNC'),('world','Geometry')]+([('C32_new_guard','PNC')]if pool=='H3'else[]):
            n=pp[pp.method==new].set_index('scene_id');o=pp[pp.method==old].set_index('scene_id').loc[n.index];delta=n.U-o.U;dv=100*delta.groupby(n.drive).mean();events={}
            for m in ['NC','DAC','DDC','TLC','TTC','HC','EC','rawNC']:
                oldevent=o[m]<1;newevent=n[m]<1;events[m]=dict(old=int(oldevent.sum()),new=int(newevent.sum()),fixed=list(n.index[oldevent&~newevent]),added=list(n.index[~oldevent&newevent]))
            comparisons[pool+':'+new+'-'+old]=dict(U=interval(dv.values),deltaP=macro((n.P-o.P).to_numpy(),n.drive.to_numpy()),events=events,changed=int((n.candidate!=o.candidate).sum()),per_drive=dv.to_dict(),top_scenes=[dict(scene_id=s,delta_U=float(v*100))for s,v in delta.reindex(delta.abs().sort_values(ascending=False).index).head(8).items()])
    write(out/'result.json',dict(complete=True,created_utc=now(),summary=summary,comparisons=comparisons,ranking={pool+':'+method:dict(pair_accuracy=float(g.correct.sum()/g.pairs.sum())if g.pairs.sum()else None,nontrivial_queries=int(g.rank_defined.sum()),mean_spearman=float(g.spearman.mean()),mean_tau_b=float(g.kendall_tau_b.mean()),macro_MAE=100*macro(g.MAE.to_numpy(),g.drive.to_numpy()))for (pool,method),g in rd.groupby(['pool','method'])},definition='drive-equal conditional single-window U16 including common prior EC; not full EPDMS; complete valid candidate labels',full_pool_oracle_label='label opportunity, not learnability guarantee'));print('R15_MATCHED_RESULT',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--data-only',action='store_true');a=p.parse_args();dataset()if a.data_only else fit(a.final)
