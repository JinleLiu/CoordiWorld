"""Recompose fixed saved predictions; drive-macro utility and paired descriptive CIs."""
import csv, argparse
from common import *

def csvwrite(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def macro(vals,dr):return float(np.mean([np.mean(vals[dr==d]) for d in np.unique(dr)]))
def evaluate(z,scores,pool):
    records=[]
    for sid in sorted(set(z['scene_id'])):
        ix=np.flatnonzero(z['scene_id']==sid)
        if pool=='M12':ix=ix[z['candidate'][ix]<12]
        ix=ix[np.argsort(z['candidate'][ix],kind='stable')]
        pick=ix[np.argmax(scores[ix])];t=z['U'][ix];s=scores[ix]
        dy=t[:,None]-t;dp=s[:,None]-s;mask=np.triu(np.ones_like(dy,dtype=bool),1);nz=mask&(dy!=0)
        r={'scene':str(sid),'drive':str(z['drive'][pick]),'fold':int(z['fold'][pick]),
           'candidate':int(z['candidate'][pick]),'row':int(pick),'n':len(ix),'U':float(z['U'][pick]),
           'oracle':float(t.max()),'regret':float(t.max()-z['U'][pick]),'predicted':float(scores[pick]),
           'pair_MSE':float(np.mean((dy[mask]-dp[mask])**2)) if mask.any() else 0.,
           'pair_MAE':float(np.mean(abs(dy[mask]-dp[mask]))) if mask.any() else 0.,
           'pair_count':int(nz.sum()),'pair_correct':float(((dy[nz]*dp[nz]>0)+.5*(dp[nz]==0)).sum())}
        for k,v in zip(['NC','DAC','DDC','TLC','P','TTC','LK','HC'],z['components'][pick]):r[k]=float(v)
        for k,v in zip(['rawNC','rawDAC','rawDDC','rawTLC','rawP','rawTTC','rawLK','rawHC'],z['raw'][pick]):r[k]=float(v)
        r['EC']=float(z['EC'][pick]);r['any_rule']=int(min(r['DAC'],r['DDC'],r['TLC'])<1)
        records.append(r)
    drives=sorted(set(r['drive'] for r in records));by=[]
    for d in drives:
        rr=[r for r in records if r['drive']==d];row={'drive':d,'scenes':len(rr)}
        for k in ['U','oracle','regret','P','TTC','LK','HC','EC','pair_MSE','pair_MAE']:row[k]=float(np.mean([r[k] for r in rr]))
        for k in ['NC','rawNC','DAC','DDC','TLC','EC','HC','TTC']:row[k+'_events']=sum(r[k]<1 for r in rr)
        row['any_rule_events']=sum(r['any_rule'] for r in rr);by.append(row)
    summary={'scenes':len(records),'drives':len(drives),'candidate_rows':sum(r['n'] for r in records)}
    for k in ['U','oracle','regret','P','TTC','LK','HC','EC','pair_MSE','pair_MAE']:summary[k]=float(np.mean([r[k] for r in by]))
    for k in ['NC','rawNC','DAC','DDC','TLC','EC','HC','TTC']:summary[k+'_events']=sum(r[k]<1 for r in records)
    summary['any_rule_events']=sum(r['any_rule'] for r in records)
    summary['pair_accuracy_pooled']=sum(r['pair_correct'] for r in records)/sum(r['pair_count'] for r in records)
    return summary,records,by
def compare(n,o):
    assert [r['scene'] for r in n]==[r['scene'] for r in o]
    delta=np.array([a['U']-b['U'] for a,b in zip(n,o)]);dr=np.array([r['drive'] for r in n]);names=np.unique(dr)
    dv=np.array([100*delta[dr==d].mean() for d in names]);rng=np.random.default_rng(20261030)
    boot=dv[rng.integers(0,len(dv),(10000,len(dv)))].mean(1)
    out={'delta_U_points':float(dv.mean()),'descriptive_drive_ci95':np.quantile(boot,[.025,.975]).tolist(),
         'positive_drives':int((dv>1e-12).sum()),'negative_drives':int((dv< -1e-12).sum()),
         'zero_drives':int((abs(dv)<=1e-12).sum()),'leave_one_drive_out_points':[(float((dv.sum()-v)/(len(dv)-1))) for v in dv],
         'changed_choices':sum(a['candidate']!=b['candidate'] for a,b in zip(n,o)),
         'by_drive':dict(zip(names.tolist(),dv.tolist()))}
    for k in ['NC','rawNC','DAC','DDC','TLC','TTC','HC','EC','any_rule']:
        event=lambda r:(bool(r[k]) if k=='any_rule' else r[k]<1)
        out[k]={'new':[a['scene'] for a,b in zip(n,o) if event(a) and not event(b)],
                'fixed':[a['scene'] for a,b in zip(n,o) if not event(a) and event(b)]}
    return out
def main():
    arg=argparse.ArgumentParser();arg.add_argument('--include-custom',action='store_true');a=arg.parse_args()
    z=dataset();ys=targets(z);y=np.c_[ys['q'],ys['qp'],ys['qttc'],ys['qlk']];c=coefficients(z)
    assert np.max(abs((y*c).sum(1)-z['U']))<1e-12
    ct=read(DATA/'experiments/structured/context.npz');assert np.array_equal(ct['HC'],z['components'][:,7])
    oldj=read(DATA/'experiments/structured/oof/predictions.npz');oldd=read(DATA/'experiments/matched/v02/oof/predictions.npz')
    summary={};choices={};drive_rows=[];pred={};health={};prefix={};comparisons={}
    allw=weights(z,np.arange(len(y)))
    for view in ['world','current']:
        n=read(ROOT/f'results/oof/{view}/predictions.npz');j=np.stack([oldj[view+'_'+h] for h in ['q','qp','qttc','qlk']],axis=1)
        f=np.stack([oldj[view+'_q'],n['p'],n['ttc'],n['lk']],axis=1)
        assert np.array_equal(np.clip(j[:,0],0,1),np.clip(f[:,0],0,1))
        ss=readouts(z,j,f,oldd[view],n['u600']);moments={'Jraw':j,'Jbox':np.clip(j,0,1),'Jproj':project(j)}
        if a.include_custom:
            custom=read(ROOT/f'results/custom/{view}/predictions.npz')
            for lam in [0,1]:
                jc=j.copy();jc[:,1]=custom[f'lambda{lam}'];assert np.array_equal(jc[:,[0,2,3]],j[:,[0,2,3]])
                for mode,mat in [('box',np.clip(jc,0,1)),('proj',project(jc))]:
                    key=f'Jlambda{lam}_{mode}';ss[key]=(mat*c).sum(1);moments[key]=mat
        for method,mat in moments.items():
            err=((mat-y)**2);invalid=(mat[:,0]<0)|(mat[:,0]>1)|(mat[:,1:]<0).any(1)|(mat[:,1:]>mat[:,0,None]+1e-12).any(1)
            health[view+':'+method]={'invalid_rows':int(invalid.sum()),'invalid_fraction':float(invalid.mean()),
                 'moment_MSE_unweighted':float(err.mean()),'moment_MSE_drive_scene_weighted':float(np.average(err.mean(1),weights=allw)),
                 'utility_MSE_unweighted':float(np.mean((ss[method]-z['U'])**2)),
                 'utility_MSE_drive_scene_weighted':float(np.average((ss[method]-z['U'])**2,weights=allw))}
        for method in ['F4','D150','D600']:
            health[view+':'+method]={'utility_MSE_unweighted':float(np.mean((ss[method]-z['U'])**2)),'utility_MSE_drive_scene_weighted':float(np.average((ss[method]-z['U'])**2,weights=allw))}
        pr=np.empty(len(y))
        for fold in range(3):
            ho=read(DATA/f'experiments/matched/v02/oof/fold{fold}/layout.npz')['hold']
            pr[ho]=np.load(ROOT/f'models/oof/{view}/fold{fold}/u600.prefix150.npy',allow_pickle=False)
        prefix[view]={'max_abs_error':float(np.max(abs(pr-oldd[view]))),'different_cells':int(np.sum(pr!=oldd[view])),'cells':len(pr)}
        pred.update({view+'_'+k:v for k,v in ss.items()});pred[view+'_F4_heads']=f;pred[view+'_J4_heads']=j
        for pool in ['M12','H3']:
            for method,scores in ss.items():
                key=f'{pool}:{view}:{method}';sm,rr,dd=evaluate(z,scores,pool);summary[key]=sm;choices[key]=rr
                drive_rows.extend([dict(pool=pool,view=view,method=method,**r) for r in dd])
            pairs=[('Jbox','F4'),('Jproj','Jbox'),('Jbox','D600'),('Jproj','D600'),('D600','D150'),('Jproj','D150')]
            if a.include_custom:pairs += [('Jlambda1_box','Jlambda0_box'),('Jlambda1_proj','Jlambda0_proj'),('Jlambda0_box','Jbox')]
            for new,old in pairs:comparisons[f'{pool}:{view}:{new}-{old}']=compare(choices[f'{pool}:{view}:{new}'],choices[f'{pool}:{view}:{old}'])
    out=ROOT/'results'/('analysis-custom' if a.include_custom else 'analysis-main');out.mkdir(parents=True,exist_ok=True)
    rows=[dict(pool=k.split(':')[0],view=k.split(':')[1],method=k.split(':')[2],**r) for k,rr in choices.items() for r in rr]
    csvwrite(out/'selected_scenes.csv',rows);csvwrite(out/'per_drive.csv',drive_rows)
    csvwrite(out/'summary.csv',[dict(pool=k.split(':')[0],view=k.split(':')[1],method=k.split(':')[2],**v) for k,v in summary.items()])
    np.savez_compressed(out/'candidate_scores.npz',**pred)
    savejson(out/'choices.json',choices);savejson(out/'results.json',{'summary':summary,'comparisons':comparisons,'health':health,'inherited_D150_vs_new_D600_prefix150':prefix,'scope':'Conditional TRAIN240 drive-OOF readout experiment. Frozen upstream features; not end-to-end independent confirmation or full two-stage EPDMS. Descriptive intervals not multiplicity-adjusted.'})
    print(json.dumps({'H3_world':{k:v['U']*100 for k,v in summary.items() if k.startswith('H3:world')},'primary_comparisons':{k:v for k,v in comparisons.items() if k.startswith('H3:world')},'prefix':prefix},indent=2))
if __name__=='__main__':main()
