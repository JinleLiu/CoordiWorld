"""Frozen-selection raw-NC / filtered-DAC proper-score evaluation, no calibration fit."""
from revision_v5_twentyfifth_common import *
import numpy as np,pandas as pd,time,argparse
from revision_v5_twentyfifth_matched import weights,interval
STUDY=BASE/'B36_original'

def register():
    p=STUDY/'PROBABILITY_PROTOCOL.json'
    if p.exists():return json.load(open(p))
    z=np.load(BASE/'train/matched_dataset_v02.npz');records=json.load(open(BASE/'train/matched_scene_records_v02.json'));ix=[]
    for r in records:
        k=r['choices']['H3']['PNC'];ii=np.flatnonzero((z['scene_id']==r['scene_id'])&(z['candidate']==k));assert len(ii)==1;ix.append(ii[0])
    ix=np.array(ix);w=weights(z['scene_id'],z['drive']);constants={}
    for name,y in [('rawNC',(z['raw'][:,0]<1).astype(float)),('DAC',(z['components'][:,1]<1).astype(float))]:
        constants[name]=dict(selected_TRAIN240=float(pd.Series(y[ix]).groupby(z['drive'][ix]).mean().mean()),all_pool_TRAIN240=float(np.average(y,weights=w)),selected_positives=int(y[ix].sum()),all_pool_positives=int(y.sum()))
    r=dict(created_utc=now(),code_sha=sha(__file__),policy='frozen R14 PNC_H3P, same selected population for all predictors',constants=constants,constant_source_sha=sha(BASE/'train/matched_dataset_v02.npz'),B36_outcomes_read=False,threshold=.5,bins=[0,.01,.05,.1,.25,.5,.75,1],weighting='parent population drive equal; selected-H3 subset retains parent weights then renormalizes',scope='proper scores for raw simulator NC<1 and human-filtered DAC<1; not real-world collision probability; no recalibration or selection change',bootstrap='10000 paired drive means seed102819 for whole population model-minus-selectedTRAINconstant Brier')
    write(p,r);return r

def scores(p,y,w):
    p=np.asarray(p,float);y=np.asarray(y,float);w=np.asarray(w,float);w=w/w.sum();pp=np.clip(p,1e-12,1-1e-12);sq=(p-y)**2;ll=-y*np.log(pp)-(1-y)*np.log1p(-pp);r=dict(n=len(y),positives=int(y.sum()),weighted_prevalence=float(w@y),Brier=float(w@sq),NLL=float(w@ll),false_negative=int(((p<=.5)&(y>0)).sum()),false_positive=int(((p>.5)&(y==0)).sum()))
    for name,mask in [('positive',y>0),('negative',y==0)]:
        r[name]=dict(n=int(mask.sum()),parent_weight_mass=float(w[mask].sum()),Brier_conditional=float(np.average(sq[mask],weights=w[mask]))if mask.any()else None,Brier_contribution=float(w[mask]@sq[mask]),NLL_contribution=float(w[mask]@ll[mask]))
    return r

def analyze(watch=False):
    reg=register()
    while not(STUDY/'analysis/result.json').exists():
        if not watch:raise RuntimeError('All1280 must finish before reading outcomes')
        check_time(2400);time.sleep(30)
    from revision_v5_twentythird_readout import Readouts
    model=None;items=[]
    for r in json.load(open(STUDY/'manifest.json'))['rows']:
        sid=r['scene_id'];d=STUDY/'scenes'/sid;record=json.load(open(d/'record.json'));k=record['choices']['PNC_H3P']
        if record['history_available']:
            a=np.load(d/'policy/PNC_H3P/features.npz');fm=record['history_contracts']['PNC_H3P']['feature_to_bank'];fi=[int(i)for i,b in fm.items()if b==k];assert len(fi)==1;fi=fi[0];pn=float(a['PNC_raw_physical_nc'][fi]);pdac=float(a['PNC_raw_dac_event'][fi])
        else:
            if model is None:model=Readouts()
            a=dict(np.load(STUDY/'base'/sid/'features.npz'));pr=model.predict(a)['PNC'];assert int(pr['choice'])==k;pn=float(pr['raw_physical_nc'][k]);pdac=float(pr['raw_dac_event'][k])
        label=json.load(open(d/'labels'/f'candidate_{k}_physical.json'));items.append(dict(scene_id=sid,drive=r['drive'],candidate=k,extra=k>=12,p_rawNC=pn,p_DAC=pdac,rawNC_event=int(label['calls'][0]['raw']['no_at_fault_collisions']<1),DAC_event=int(label['filtered']['drivable_area_compliance']<1)))
    df=pd.DataFrame(items);counts=df.drive.value_counts();df['parent_weight']=[1/counts[d]for d in df.drive];out=STUDY/'probability';out.mkdir(exist_ok=True);df.to_csv(out/'per_scene.csv',index=False);result={};bins=[]
    for event in ['rawNC','DAC']:
        result[event]={};p=df['p_'+event].to_numpy();y=df[event+'_event'].to_numpy();w=df.parent_weight.to_numpy()
        for name,mask in [('all',np.ones(len(df),bool)),('selected_H3',df.extra.to_numpy()),('selected_M12',~df.extra.to_numpy())]:
            pred={'model':p,'constant_selected_TRAIN240':np.full(len(p),reg['constants'][event]['selected_TRAIN240']),'constant_all_pool_TRAIN240':np.full(len(p),reg['constants'][event]['all_pool_TRAIN240'])};result[event][name]={n:scores(pp[mask],y[mask],w[mask])for n,pp in pred.items()}
        c=reg['constants'][event]['selected_TRAIN240'];delta=pd.Series((p-y)**2-(c-y)**2).groupby(df.drive).mean();result[event]['Brier_model_minus_selectedTRAINconstant']=interval(delta.values)
        for lo,hi in zip(reg['bins'][:-1],reg['bins'][1:]):
            m=(p>=lo)&((p<=hi)if hi==1 else(p<hi));bins.append(dict(event=event,lo=lo,hi=hi,n=int(m.sum()),positives=int(y[m].sum()),mean_prediction=float(np.average(p[m],weights=w[m]))if m.any()else None,event_rate=float(np.average(y[m],weights=w[m]))if m.any()else None))
    pd.DataFrame(bins).to_csv(out/'reliability_bins.csv',index=False);write(out/'result.json',dict(complete=True,created_utc=now(),protocol_sha=sha(STUDY/'PROBABILITY_PROTOCOL.json'),code_sha=sha(__file__),summary=result,no_selection_or_fit_change=True));print('R15_B36_PROBABILITY_COMPLETE',result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args();register()if a.register else analyze(a.watch)
