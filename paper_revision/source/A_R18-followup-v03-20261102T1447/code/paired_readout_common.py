"""Shared deployable inputs, guard and switch rule for matched P and delta-U."""
import numpy as np


def utility(z):
    c = z['components']
    return np.prod(c[...,:4],axis=-1)*(5*c[...,4]+5*c[...,5]+2*c[...,6]+2*c[...,7]+2*z['EC'])/16


def candidate_features(z):
    return np.concatenate([z['x674'],z['C32'],z['M56'],z['EC_proxy'][...,None],
                           z['features198'][...,197,None]],axis=-1).astype(np.float32)


def paired_features(x, baseline):
    xb = x[np.arange(len(x)),baseline][:,None,:]
    out = np.concatenate([x,np.broadcast_to(xb,x.shape),x-xb],axis=-1)
    # Preserve any missing feature as NaN. Infinite features are an interface error.
    assert not np.isinf(out).any()
    return out.astype(np.float32)


def common_guard(z):
    rr=np.arange(len(z['old_choice']));b=z['old_choice']
    risk=z['old_physical_nc'];dac=z['old_dac_event']
    guard=(z['valid'] & z['old_eligible'] &
           (risk<=np.maximum(.5,risk[rr,b])[:,None]) &
           (dac<=np.maximum(.5,dac[rr,b])[:,None]))
    assert guard[rr,b].all()
    return guard


def choose(arm, predictions, baseline, guard, margin):
    rr=np.arange(len(baseline))
    gains=predictions.copy()
    if arm=='P':gains-=predictions[rr,baseline][:,None]
    else:assert arm=='D'
    gains[rr,baseline]=0.0
    assert np.isfinite(gains[guard]).all()
    best=np.where(guard,gains,-np.inf).argmax(1)
    selected=np.where(gains[rr,best]>margin,best,baseline)
    return selected,gains


def training_rows(drives, valid, scenes):
    rr,cc=np.nonzero(valid[scenes]);global_r=scenes[rr]
    counts={d:int((drives[scenes]==d).sum()) for d in set(drives[scenes])}
    w=np.array([1/(counts[drives[i]]*valid[i].sum()) for i in global_r],dtype=np.float64)
    w/=w.mean()
    return global_r,cc,w


def evaluate(z, choices, inventory):
    u=utility(z);valid=z['valid'];complete=(np.isfinite(u)|~valid).all(1)
    rr=np.arange(len(valid));b=z['old_choice'];base=u[rr,b];comp=z['components']
    methods={'M0':b,**choices};summaries={};scene_rows=[];drive_rows=[]
    for i in rr:
        for arm,ch in methods.items():
            j=int(ch[i]);c=comp[i,j]
            row=dict(method=arm,scene_id=str(z['scene_ids'][i]),drive=str(z['drives'][i]),fold=int(z['fold'][i]),
                regression240=bool(z['refreshed'][i]),alignment_status='AVAILABLE',missing_reason=None,
                U16_complete=bool(complete[i]),candidate=j,candidate_key=str(z['candidate_key'][i,j]),baseline_candidate=int(b[i]),
                changed=bool(j!=b[i]),U16=float(u[i,j]) if complete[i] else None,
                delta_vs_M0_pp=float(100*(u[i,j]-base[i])) if complete[i] else None,
                rawNC=float(z['rawNC'][i,j]),filteredNC=float(c[0]),DAC=float(c[1]),DDC=float(c[2]),TLC=float(c[3]),
                EP=float(c[4]),TTC=float(c[5]),LK=float(c[6]),HC=float(c[7]),EC=float(z['EC'][i,j]) if np.isfinite(z['EC'][i,j]) else None)
            scene_rows.append(row)
    for missing in inventory['rows']:
        if missing['alignment_status']=='AVAILABLE':continue
        for arm in methods:
            row=dict.fromkeys(scene_rows[0]);row.update({k:missing[k] for k in ['scene_id','drive','fold','regression240','alignment_status','missing_reason']})
            row.update(method=arm,U16_complete=False);scene_rows.append(row)
    order={x['scene_id']:i for i,x in enumerate(inventory['rows'])}
    scene_rows.sort(key=lambda x:(order[x['scene_id']],list(methods).index(x['method'])))
    for group,gmask in {'natural1755':np.ones(len(valid),bool),'old_exposed240':z['refreshed'],'remaining1515':~z['refreshed']}.items():
        mask=gmask & complete;drives=sorted(set(z['drives'][mask]));component_drives=sorted(set(z['drives'][gmask]))
        requested=[x for x in inventory['rows'] if group=='natural1755' or x['regression240']==(group=='old_exposed240')]
        def dm(values):return np.array([np.mean(values[mask & (z['drives']==d)]) for d in drives])
        summaries[group]={}
        for arm,ch in methods.items():
            c=comp[rr,ch];bc=comp[rr,b];raw=z['rawNC'][rr,ch];br=z['rawNC'][rr,b]
            delta=100*dm(u[rr,ch]-base);ep=np.array([(c[:,4]-bc[:,4])[gmask & (z['drives']==d)].mean() for d in component_drives])
            events={}
            for name,old,new in [('rawNC',br,raw),('filteredNC',bc[:,0],c[:,0]),('DAC',bc[:,1],c[:,1]),
                                 ('DDC',bc[:,2],c[:,2]),('TLC',bc[:,3],c[:,3]),('TTC',bc[:,5],c[:,5]),
                                 ('HC',bc[:,7],c[:,7]),('EC',z['EC'][rr,b],z['EC'][rr,ch])]:
                em=gmask & np.isfinite(old) & np.isfinite(new);bad0=old<1;bad=new<1
                events[name]=dict(denominator=int(em.sum()),old=int((em & bad0).sum()),new=int((em & bad).sum()),
                    added=list(map(str,z['scene_ids'][em & ~bad0 & bad])),repaired=list(map(str,z['scene_ids'][em & bad0 & ~bad])))
            boot=delta[np.random.default_rng(170018).integers(0,len(drives),size=(10000,len(drives)))].mean(1)
            gate=bool(arm!='M0' and delta.mean()>1e-9 and (delta>1e-9).sum()>=(delta< -1e-9).sum() and ep.mean()>=-.01
                and all(events[k]['new']<=events[k]['old'] for k in ['rawNC','filteredNC','DAC','DDC','TLC']))
            summaries[group][arm]=dict(requested_scenes=len(requested),available_current_scenes=int(gmask.sum()),complete_U16_scenes=int(mask.sum()),
                drives=len(drives),drive_macro_Score=float(100*dm(u[rr,ch]).mean()),scene_mean_Score=float(100*u[rr,ch][mask].mean()),
                paired_delta_pp=float(delta.mean()),descriptive_drive_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                improved_drives=int((delta>1e-9).sum()),worsened_drives=int((delta< -1e-9).sum()),unchanged_drives=int((abs(delta)<=1e-9).sum()),
                changed_choices=int((gmask & (ch!=b)).sum()),delta_EP=float(ep.mean()),events=events,development_gate=gate)
            for d,du,score in zip(drives,delta,100*dm(u[rr,ch])):
                drive_rows.append(dict(group=group,method=arm,drive=str(d),Score=float(score),delta_vs_M0_pp=float(du),
                    complete_U16_scenes=int((mask & (z['drives']==d)).sum()),requested_scenes=sum(x['drive']==str(d) for x in requested)))
    assert len(scene_rows)==1755*len(methods)
    return summaries,scene_rows,drive_rows
