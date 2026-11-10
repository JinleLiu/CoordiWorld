"""Two prospectively fixed, head-independent comparators on shared features.

Only current/frozen-world features and deployable controller EC enter this file.
No labels, true/native EC, models, learned eligibility or CW head values are read.
All fields retain the inherited producer's scales. Missing support is explicit.
"""
import numpy as np

RULES=('A_lexicographic','B_fixed_utility')
INPUT_FIELDS=('candidates','valid','features198','x674','C32','R8')

def proxies(a, ec_proxy):
    f=np.asarray(a['features198'],float);x=np.asarray(a['x674'],float)
    c=np.asarray(a['C32'],float);road=np.asarray(a['R8'],float)
    n=len(a['valid']);valid=np.asarray(a['valid'],bool).copy()
    valid &= np.isfinite(a['candidates']).all((1,2))
    assert f.shape==(n,198) and x.shape==(n,674) and c.shape==(n,32) and road.shape==(n,8)
    assert all(np.isfinite(v[valid]).all() for v in [f,x,c,road])
    assert valid.any()
    # C32 actual semantic contact proxies; SAT margins are not probabilities.
    contact=np.maximum(c[:,0],c[:,1])>0
    # features198[120:184] contains 8 x [four current-rule margins, four support flags].
    # Channels: lane, drivable, stop, signal. Margins have scale metres / 10.
    rules=f[:,120:184].reshape(n,8,8);margin=rules[:,:,:4];known=rules[:,:,4:]>0.5
    violations=(margin<0)&known
    control_bad=violations[:,:,2:].any((1,2))
    control_known=known[:,:,2:].any((1,2))
    dac_known=(f[:,191]>.5)|(road[:,0]>0)|known[:,:,1].any(1)
    dac_bad=((f[:,191]>.5)&(f[:,192]<1))|(road[:,2]>0)|violations[:,:,1].any(1)
    ddc_known=f[:,193]>.5;ddc_bad=ddc_known&(f[:,194]<1)
    # Corrected S64 contains four current/frozen-world TTC projection contact fractions.
    ttc_bad=(x[:,594+np.array([3,7,11,15])]>0).any(1)
    ttc=1-ttc_bad.astype(float)
    lane_known=known[:,:,0].any(1)
    lk=np.divide((known[:,:,0]&~violations[:,:,0]).sum(1),known[:,:,0].sum(1),out=np.zeros(n),where=known[:,:,0].sum(1)>0)
    progress_known=(x[:,668]>.5)|(x[:,669]>.5)
    progress=np.clip(x[:,662],0,1);distance=np.where(x[:,668]>.5,100*x[:,658],0)
    hc=np.clip(f[:,197],0,1)
    ec=np.full(n,np.nan) if ec_proxy is None else np.asarray(ec_proxy,float)
    assert ec.shape==(n,) and not np.isinf(ec).any()
    ec_known=np.isfinite(ec);ec_value=np.where(ec_known,np.clip(ec,0,1),0)
    weights=5*progress_known+5+2*lane_known+2+2*ec_known
    core=(5*progress*progress_known+5*ttc+2*lk*lane_known+2*hc+2*ec_value)/weights
    # Unknown multiplicative rules are omitted, not reported as observed-safe.
    product=(~contact)&(~dac_bad)&(~ddc_bad)&(~control_bad)
    utility=product*core
    rule_cost=dac_bad.astype(int)+ddc_bad.astype(int)+control_bad.astype(int)
    return dict(valid=valid,NC_proxy=1-contact.astype(float),DAC_proxy=np.where(dac_known,1-dac_bad.astype(float),np.nan),
        DDC_proxy=np.where(ddc_known,1-ddc_bad.astype(float),np.nan),control_proxy=np.where(control_known,1-control_bad.astype(float),np.nan),
        TTC_proxy=ttc,LK_proxy=np.where(lane_known,lk,np.nan),EP_proxy=np.where(progress_known,progress,np.nan),HC_proxy=hc,EC_proxy=ec,
        contact=contact,rule_cost=rule_cost,ttc_bad=ttc_bad,core=core,utility=utility,distance=distance,
        progress_known=progress_known,control_known=control_known,lane_known=lane_known,
        support_missing_count=(~dac_known).astype(int)+(~ddc_known)+(~control_known)+(~progress_known)+(~lane_known)+(~ec_known))

def select(a,ec_proxy,rule,unsafe=None):
    assert rule in RULES
    p=proxies(a,ec_proxy);valid=p['valid'];ids=np.flatnonzero(valid)
    # Stable slot ID is the final tie-break; all candidates share the same rule.
    if rule==RULES[0]:
        keys=lambda j:(int(p['contact'][j]),int(p['rule_cost'][j]),int(p['ttc_bad'][j]),-p['core'][j],-p['distance'][j],int(j))
    else:
        keys=lambda j:(-p['utility'][j],int(p['contact'][j])+int(p['rule_cost'][j]),-p['core'][j],-p['distance'][j],int(j))
    order=np.array(sorted(ids,key=keys),int);before=int(order[0]);chosen=before
    if unsafe is not None:
        unsafe=np.asarray(unsafe,bool);assert unsafe.shape==valid.shape
        clear=order[~unsafe[order]]
        # Identical historical staticGuard operator: change only an unsafe winner
        # if at least one eligible clear alternative exists, otherwise keep winner.
        if unsafe[before] and len(clear):chosen=int(clear[0])
    return chosen,dict(proxies=p,ordering=order,pre_guard_choice=before,guard_changed=chosen!=before)

def guard_from_native(native,current,valid,guard_function,vehicle):
    """Existing staticGuard input adapter; current observations only."""
    origin=np.asarray(current['rear_axle_pose'],float);co,si=np.cos(origin[2]),np.sin(origin[2])
    local=np.asarray(native[...,:3],float).copy()
    local[...,:2]=(native[...,:2]-origin[:2])@np.array([[co,-si],[si,co]])
    local[...,2]=np.arctan2(np.sin(native[...,2]-origin[2]),np.cos(native[...,2]-origin[2]))
    return guard_function(local,np.linalg.norm(native[...,3:5],axis=-1),np.asarray(current['states']),np.asarray(current['types']),
        np.ones(len(current['states']),bool),np.asarray(valid,bool),[vehicle.length,vehicle.width],vehicle.rear_axle_to_center)
