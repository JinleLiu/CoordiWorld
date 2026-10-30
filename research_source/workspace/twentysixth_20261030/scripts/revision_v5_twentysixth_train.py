"""Matched scalar-U16 C/CV/Hybrid: one budget, full common H3 training pool."""
from revision_v5_twentysixth_common import *
import argparse,time
import numpy as np,pandas as pd,xgboost as xgb
from scipy.stats import spearmanr,kendalltau
from revision_v5_twentyfifth_matched import PARAMS,weights,macro,interval,SHORT
ARMS=['current','cv','hybrid']

def fit(final=False):
    z=dict(np.load(BASE/'train/dataset.npz'));out=BASE/'models'/('final'if final else'oof');out.mkdir(parents=True,exist_ok=True)
    assert xgb.__version__=='2.1.4'
    pred={a:np.full(len(z['U']),np.nan)for a in ARMS};receipts=[];start=time.monotonic()
    for fold in ([None]if final else range(3)):
        check_time();train=np.arange(len(z['U']))if fold is None else np.flatnonzero(z['fold']!=fold);hold=np.arange(len(z['U']))if fold is None else np.flatnonzero(z['fold']==fold)
        train=train[np.lexsort((z['candidate'][train],z['scene_id'][train]))];w=weights(z['scene_id'][train],z['drive'][train]);d=out/('fit'if fold is None else'f'+str(fold));d.mkdir(exist_ok=True)
        np.savez_compressed(d/'layout.npz',train=train,hold=hold,weights=w);record={}
        assert final or set(z['drive'][train]).isdisjoint(set(z['drive'][hold]))
        for arm in ARMS:
            tt=time.perf_counter();m=xgb.train(PARAMS,xgb.DMatrix(z[arm][train],label=z['U'][train],weight=w,nthread=1),num_boost_round=150,verbose_eval=False);fitseconds=time.perf_counter()-tt
            m.save_model(d/(arm+'.ubj'));write(d/(arm+'_config.json'),json.loads(m.save_config()))
            tt=time.perf_counter();pred[arm][hold]=m.inplace_predict(z[arm][hold]);inference=time.perf_counter()-tt;fp=m.inplace_predict(z[arm][train]);np.savez_compressed(d/(arm+'_fit.npz'),indices=train,predictions=fp)
            tree=m.trees_to_dataframe();record[arm]=dict(model_sha=sha(d/(arm+'.ubj')),model_bytes=(d/(arm+'.ubj')).stat().st_size,dimensions=m.num_features(),trees=m.num_boosted_rounds(),leaves=int((tree.Feature=='Leaf').sum()),fit_seconds=fitseconds,one_batch_predict_seconds=inference,predict_rows=len(hold),fit_weighted_MSE=float(np.average((fp-z['U'][train])**2,weights=w)))
            print('R16_FIT',fold,arm,round(fitseconds,3),record[arm]['leaves'],flush=True)
        write(d/'receipt.json',record);receipts.append(dict(fold=fold,arms=record))
    assert all(np.isfinite(v).all()for v in pred.values())
    np.savez_compressed(out/'predictions.npz',**pred)
    old=np.load(B25/'matched/v02'/('final'if final else'oof')/'predictions.npz')
    current_err=float(abs(pred['current']-old['current']).max());assert current_err==0,current_err
    write(out/'fit_complete.json',dict(complete=True,created_utc=now(),seconds=time.monotonic()-start,receipts=receipts,source_sha=sha(__file__),dataset_sha=sha(BASE/'train/dataset.npz'),params=PARAMS,rounds=150,xgboost=xgb.__version__,current_R15_prediction_maxerror=current_err,final_train_in_sample=final))
    analyze(z,pred,out)

def analyze(z,pred,out):
    recs=json.load(open(BASE/'train/records.json'));rows=[];rank=[]
    for r in recs:
        ii=np.arange(r['row_start'],r['row_start']+r['row_count']);lookup=dict(zip(z['candidate'][ii],ii))
        for pool in ['M12','H3']:
            ids=ii[z['candidate'][ii]<12]if pool=='M12'else ii;oracle=float(z['U'][ids].max());choices={m:lookup[k]for m,k in r['choices'][pool].items()if m in ['PNC','IC0','Geometry']}
            choices.update({m:int(ids[np.argmax(p[ids])])for m,p in pred.items()})
            for method,k in choices.items():
                rows.append(dict(pool=pool,method=method,scene_id=r['scene_id'],drive=r['drive'],fold=r['fold'],candidate=int(z['candidate'][k]),family=str(z['family'][k]),U=float(z['U'][k]),EC=float(z['EC'][k]),oracle_U=oracle,regret=oracle-float(z['U'][k]),**dict(zip(SHORT,z['components'][k].tolist())),**{'raw'+m:float(v)for m,v in zip(SHORT,z['raw'][k])}))
            for method,p in pred.items():
                y=z['U'][ids];s=p[ids];dy=y[:,None]-y;ds=s[:,None]-s;mask=np.triu(dy!=0,1);valid=len(ids)>1 and np.ptp(y)>0 and np.ptp(s)>0
                rank.append(dict(pool=pool,method=method,scene_id=r['scene_id'],drive=r['drive'],pairs=int(mask.sum()),correct=float(((dy[mask]*ds[mask]>0)+.5*(ds[mask]==0)).sum()),MSE=float(np.mean((s-y)**2)),MAE=float(abs(s-y).mean()),spearman=float(spearmanr(y,s).statistic)if valid else None,kendall_tau_b=float(kendalltau(y,s).statistic)if valid else None))
    df=pd.DataFrame(rows);df.to_csv(out/'per_scene.csv',index=False);rd=pd.DataFrame(rank);rd.to_csv(out/'ranking.csv',index=False)
    df.groupby(['pool','method','drive'])[['U','P','EC','regret']].mean().reset_index().to_csv(out/'per_drive.csv',index=False)
    summary={};contrasts={};eventcols=['NC','DAC','DDC','TLC','TTC','LK','HC','EC','rawNC','rawDAC','rawDDC','rawTLC']
    for pool in ['M12','H3']:
        pp=df[df.pool==pool];summary[pool]={}
        for m,g in pp.groupby('method'):
            dr=g.drive.to_numpy();summary[pool][m]=dict(U=100*macro(g.U.to_numpy(),dr),oracle_regret=100*macro(g.regret.to_numpy(),dr),scenes=len(g),drives=len(set(dr)),means={k:macro(g[k].to_numpy(),dr)for k in SHORT+['EC']},events={k:int((g[k]<1).sum())for k in eventcols},H3_selected=int((g.candidate>=12).sum()))
        for new,old in [('hybrid','cv'),('cv','current'),('hybrid','current'),('hybrid','PNC'),('cv','PNC')]:
            n=pp[pp.method==new].set_index('scene_id');o=pp[pp.method==old].set_index('scene_id').loc[n.index];delta=n.U-o.U;dv=100*delta.groupby(n.drive).mean();events={}
            for k in eventcols:
                ne=n[k]<1;oe=o[k]<1;events[k]=dict(new_total=int(ne.sum()),old_total=int(oe.sum()),fixed=list(n.index[oe&~ne]),added=list(n.index[~oe&ne]))
            contrasts[pool+':'+new+'-'+old]=dict(U=interval(dv.to_numpy()),per_drive=dv.to_dict(),deltaP=macro((n.P-o.P).to_numpy(),n.drive.to_numpy()),changed=int((n.candidate!=o.candidate).sum()),events=events)
    write(out/'result.json',dict(complete=True,created_utc=now(),summary=summary,comparisons=contrasts,ranking={p+':'+m:dict(pair_accuracy=float(g.correct.sum()/g.pairs.sum()),mean_spearman=float(g.spearman.mean()),macro_MAE=macro(g.MAE.to_numpy(),g.drive.to_numpy()),macro_MSE=macro(g.MSE.to_numpy(),g.drive.to_numpy()))for(p,m),g in rd.groupby(['pool','method'])},evaluation='TRAIN conditional on frozen upstream bank/prior; full valid labels; drive-equal U16 x100; final folder is fit-in-sample, OOF folder is held-drive predictions'))
    print('R16_RESULTS',{p:{m:round(v['U'],6)for m,v in mm.items()}for p,mm in summary.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:
        out=BASE/'models'/('final'if a.final else'oof');analyze(dict(np.load(BASE/'train/dataset.npz')),dict(np.load(out/'predictions.npz')),out)
    else:fit(a.final)
