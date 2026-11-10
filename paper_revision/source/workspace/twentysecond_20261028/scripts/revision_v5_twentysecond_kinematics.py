"""Explicit native/body vs scene-vector feature contract; no evaluator mutation.

Native pose is global. Native velocity/acceleration components are body slots.
The acceleration feature vector below is rotated body-component acceleration,
NOT inertial acceleration: it intentionally does not add v_long * yaw_rate.
Longitudinal acceleration is always native slot5, including observed t0.
"""
import numpy as np
def from_native(native,origin):
    origin=np.asarray(origin,float);a=np.asarray(native,float);v=a.copy();co,si=np.cos(origin[2]),np.sin(origin[2]);v[...,:2]=(a[...,:2]-origin[:2])@np.array([[co,-si],[si,co]]);yaw=np.arctan2(np.sin(a[...,2]-origin[2]),np.cos(a[...,2]-origin[2]));v[...,2]=yaw;c,s=np.cos(yaw),np.sin(yaw)
    for lo in [3,5]:v[...,lo]=c*a[...,lo]-s*a[...,lo+1];v[...,lo+1]=s*a[...,lo]+c*a[...,lo+1]
    return dict(execution_native=a.copy(),execution_feature_scene=v,acceleration_longitudinal=a[...,5].copy(),velocity_body=a[...,3:5].copy())
def recover_legacy(legacy,origin):
    """Invert the historic erroneous transform only to recover feature inputs.

    Official scoring/EC continue consuming original native saved states, or the
    unchanged old recovery contract. Do NOT inverse-rotate the new feature view.
    """
    o=np.asarray(origin,float);a=np.asarray(legacy,float);out=a.copy();c,s=np.cos(o[2]),np.sin(o[2]);rt=np.array([[c,s],[-s,c]]);out[...,:2]=a[...,:2]@rt+o[:2];out[...,2]=np.arctan2(np.sin(a[...,2]+o[2]),np.cos(a[...,2]+o[2]));out[...,3:5]=a[...,3:5]@rt;out[...,5:7]=a[...,5:7]@rt;return out
def views(assets,current):
    old=assets['execution_full_local'];v=from_native(recover_legacy(old,current['rear_axle_pose']),current['rear_axle_pose'])
    # Keep the unchanged local pose bytes, avoiding subtraction/angle roundoff.
    v['execution_feature_scene'][...,:3]=old[...,:3]
    c,s=np.cos(old[...,2]),np.sin(old[...,2]);a=v['execution_native']
    for lo in [3,5]:
        v['execution_feature_scene'][...,lo]=c*a[...,lo]-s*a[...,lo+1];v['execution_feature_scene'][...,lo+1]=s*a[...,lo]+c*a[...,lo+1]
    return v
def fix_s64(assets,current,old_s64,trace):
    from revision_v5_sixteenth_semantics import interpolate
    view=views(assets,current);out=old_s64.copy();ex=view['execution_feature_scene'];pred,_=interpolate(assets);nn=trace['risk_actor_index'];bb=trace['risk_base_index'];kk=trace['risk_projection_index'];ok=(nn>=0)&assets['candidate_valid'];jj=np.flatnonzero(ok)
    if len(jj):
        actor=pred[bb[jj]+np.array([0,3,6,9])[kk[jj]],nn[jj]];rel=actor[:,3:5]-ex[jj,bb[jj],3:5];yaw=ex[jj,bb[jj],2];out[jj,18]=(rel*np.c_[np.cos(yaw),np.sin(yaw)]).sum(-1);out[jj,19]=(rel*np.c_[-np.sin(yaw),np.cos(yaw)]).sum(-1)
    out[~ok,18:20]=0;return out,view
def fix_signal(old,view,valid):
    out=old.copy();a=view['acceleration_longitudinal'];out[:,5]=a[:,-1]/10;out[:,7]=np.maximum(-a.min(1),0)/10;out[~valid]=0;return out
