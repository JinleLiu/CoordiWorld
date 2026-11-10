"""Current-state continuation of an issued plan's predicted execution.

No future observation or scoring label is accepted. Native body slots retain
their meaning. The previous prediction is a causal, policy-owned plan buffer.
"""
import numpy as np

def wrap(a):return np.arctan2(np.sin(a),np.cos(a))

def quintic(p,v,a,T,t,derivative=0):
    mat=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
    tail=np.linalg.solve(mat,np.stack([-p-v*T-a*T*T/2,-v-a*T,-a]))
    coef=np.vstack([p,v,a/2,tail]);tt=np.minimum(t,T)
    for _ in range(derivative):coef=coef[1:]*np.arange(1,len(coef))[:,None]
    out=sum(tt[:,None]**k*c for k,c in enumerate(coef))
    if derivative:out[t>=T]=0
    return out

def extend(previous,offset):
    p=np.array(previous[offset:],float,copy=True);n=41-len(p);assert n>=0
    # Only the missing tail (usually .5s): decay terminal longitudinal a to0,
    # retain terminal curvature, integrate nonnegative speed in .1s steps.
    rows=list(p);v=max(0.,float(p[-1,3]));a=float(p[-1,5]);yaw=float(p[-1,2]);xy=p[-1,:2].copy();k=float(p[-1,9])/max(v,.5)
    for j in range(n):
        an=a*(1-(j+1)/max(n,1));vn=max(0.,v+.1*an);ds=.05*(v+vn);mid=yaw+.5*k*ds;xy=xy+ds*np.array([np.cos(mid),np.sin(mid)]);yaw+=k*ds;r=p[-1].copy();r[:3]=[*xy,wrap(yaw)];r[3:7]=[vn,0,an,0];r[9]=vn*k;rows.append(r);v=vn
    return np.array(rows),dict(retained_steps=len(p),extension_steps=n,extension_seconds=.1*n,extension='terminal curvature; longitudinal acceleration decays to zero; no reverse')

def continuation(current_native,previous_native,timestamp_us,previous_timestamp_us,join_seconds=3.,kind='time'):
    c=np.asarray(current_native,float);p=np.asarray(previous_native,float);elapsed=(int(timestamp_us)-int(previous_timestamp_us))/1e5;off=round(elapsed)
    if not(0<off<len(p)-2):raise ValueError('No legal overlapping prior plan')
    ref,meta=extend(p,off);theta=c[2];rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);xy=(ref[:,:2]-c[:2])@rot;heading=np.unwrap(ref[:,2]-theta);t=np.arange(41)*.1
    vref=ref[0,3]*np.array([np.cos(heading[0]),np.sin(heading[0])]);aref=ref[0,5]*np.array([np.cos(heading[0]),np.sin(heading[0])])+ref[0,3]*ref[0,9]*np.array([-np.sin(heading[0]),np.cos(heading[0])])
    vcur=np.array([c[3],c[4]]);acur=np.array([c[5],c[3]*c[9]])
    if kind=='time':
        delta=quintic(-xy[0],vcur-vref,acur-aref,join_seconds,t);path=xy+delta
        # Native positions follow a discrete bicycle integration; use numerical
        # derivatives of the actual path plus exact correction derivative.
        vel=np.gradient(xy,.1,axis=0,edge_order=2)+quintic(-xy[0],vcur-vref,acur-aref,join_seconds,t,1)
        yaw=np.arctan2(vel[:,1],vel[:,0]);yaw[np.linalg.norm(vel,axis=1)<.05]=heading[np.linalg.norm(vel,axis=1)<.05]
    elif kind=='relative':
        # Mechanism control: same dynamic profile placed at the actual origin.
        # This preserves initial pose, but does not match current v/a.
        rr=np.array([[np.cos(-heading[0]),-np.sin(-heading[0])],[np.sin(-heading[0]),np.cos(-heading[0])]])
        path=(xy-xy[0])@rr.T;yaw=heading-heading[0]
    elif kind=='dynamic':
        # Reuse the prior acceleration/curvature profile, NOT its speed or
        # absolute endpoint. Position catch-up can create extra jerk; a speed
        # difference need not be removed when reusing a legal historical plan.
        at=ref[:,5];kt=ref[:,9]/np.maximum(abs(ref[:,3]),.5)
        v=max(0.,float(c[3]));k0=float(c[9])/max(v,.5);pos=np.zeros(2);th=0.;path=[pos.copy()];yaw=[th]
        for step in range(1,401):
            tm=step*.01;u=min(tm/join_seconds,1.);fade=1-3*u*u+2*u*u*u
            ac=float(np.interp(tm,t,at))+(c[5]-at[0])*fade;kap=float(np.interp(tm,t,kt))+(k0-kt[0])*fade
            vn=max(0.,v+.01*ac);ds=.005*(v+vn);mid=th+.5*kap*ds;pos=pos+ds*np.array([np.cos(mid),np.sin(mid)]);th+=kap*ds;v=vn
            if step%10==0:path.append(pos.copy());yaw.append(th)
        path=np.array(path);yaw=np.array(yaw)
    else:raise ValueError(kind)
    path[0]=0;yaw[0]=0;dense=np.c_[path,wrap(yaw)];cand=dense[5::5].astype(np.float32)
    assert cand.shape==(8,3)and np.isfinite(cand).all()
    return cand,dict(**meta,kind=kind,join_seconds=join_seconds,offset_steps=off,elapsed_seconds=elapsed*.1,prior_position_error_m=float(np.linalg.norm(xy[0])),prior_heading_error_rad=float(wrap(heading[0])),velocity_error_mps=(vcur-vref).tolist(),acceleration_error_mps2=(acur-aref).tolist(),initial_native=c.tolist(),dense=dense.tolist())

def ec_matrix(current,previous,offset):
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_two_frame_extended_comfort
    n=min(current.shape[1],len(previous)-offset);assert n>3
    return ego_is_two_frame_extended_comfort(current[:,:n],np.broadcast_to(previous[offset:offset+n],current[:,:n].shape),np.arange(n)*.1).astype(float)
