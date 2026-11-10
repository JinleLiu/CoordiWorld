"""S64: fixed current/predicted-world semantics, no outcome labels or fitting.

SAT margins are not Euclidean distances; TTC-like eligibility is a proxy and
does not reproduce native collision-history exclusions or human filtering.
"""
import numpy as np

def sat(a,b):
    ca,sa,cb,sb=np.cos(a[...,2]),np.sin(a[...,2]),np.cos(b[...,2]),np.sin(b[...,2])
    ua,va=np.stack([ca,sa],-1),np.stack([-sa,ca],-1)
    ub,vb=np.stack([cb,sb],-1),np.stack([-sb,cb],-1);d=b[...,:2]-a[...,:2]
    return np.maximum.reduce([abs((d*v).sum(-1))-abs((ua*v).sum(-1))*a[...,3]/2-abs((va*v).sum(-1))*a[...,4]/2-abs((ub*v).sum(-1))*b[...,3]/2-abs((vb*v).sum(-1))*b[...,4]/2 for v in [ua,va,ub,vb]])

def interpolate(a):
    w=np.concatenate([a['current'][None],a['world']],0).astype(float)
    # Unwrap only valid-current paths; otherwise first prediction anchors angle.
    w[0,~a['current_known']]=w[1,~a['current_known']]
    w[:,:,2]=np.unwrap(w[:,:,2],axis=0)
    grid=np.arange(41)/5;lo=np.minimum(grid.astype(int),7);frac=grid-lo
    pred=w[lo]*(1-frac[:,None,None])+w[lo+1]*frac[:,None,None]
    pred[:,:,2]=np.arctan2(np.sin(pred[:,:,2]),np.cos(pred[:,:,2]))
    valid=np.broadcast_to(a['actor_valid'],(41,len(a['actor_valid']))).copy()
    valid[:5]&=a['current_known'][None]
    return pred,valid

def semantic_features(a,area):
    pred,pvalid=interpolate(a);ex=a['execution_full_local'];valid=a['candidate_valid'];M=len(valid);N=len(a['actor_valid'])
    assert ex.shape[1]==41 and pred.shape[0]==41
    B=32;speed=np.linalg.norm(ex[:,:B,3:5],axis=-1);yaw=ex[:,:B,2];fw=np.stack([np.cos(yaw),np.sin(yaw)],-1);lf=np.stack([-np.sin(yaw),np.cos(yaw)],-1)
    known,multi,nondrive,inter=[area[k][:,:B] for k in ['known','multiple','nondrive','intersection']]
    area_or=(multi|nondrive|inter)&known
    gaps=[];eligible=[];ahead=[];behind=[];actors=[];pv=[]
    for step in [0,3,6,9]:
        actor=pred[np.arange(B)+step];ok=pvalid[np.arange(B)+step]
        pos=ex[:,:B,:2]+fw*(float(np.asarray(a['rear_axle_offset']).ravel()[0])+speed*step/10)[...,None]
        eb=np.concatenate([pos,yaw[...,None],np.broadcast_to(a['ego_box'],(M,B,2))],-1)
        ab=np.concatenate([actor[:,:,:3],actor[:,:,5:7]],-1)
        g=sat(eb[:,:,None,:],ab[None]);delta=actor[None,:,:,:2]-ex[:,:B,None,:2]
        lon=(delta*fw[:,:,None]).sum(-1);lat=(delta*lf[:,:,None]).sum(-1)
        ang=np.abs(np.arctan2(lat,lon));ah=ang<np.pi/6;bh=ang>5*np.pi/6
        el=(ah|(area_or[:,:,None]&~bh))&(speed[:,:,None]>=.005)&ok[None]
        gaps.append(np.where(ok[None],g,np.inf));eligible.append(el);ahead.append(ah);behind.append(bh);actors.append(actor);pv.append(ok)
    g=np.stack(gaps,2);el=np.stack(eligible,2);ah=np.stack(ahead,2);bh=np.stack(behind,2)
    allok=np.stack(pv,1);f=[];names=[]
    def add(value,name):f.append(np.broadcast_to(value,(M,)).astype(float));names.append(name)
    def boundedmin(x,axis):return np.clip(np.min(x,axis=axis,initial=np.inf),-50,50)
    for k,offset in enumerate([0,.3,.6,.9]):
        contact=(el[:,:,k]&(g[:,:,k]<=0)).any(-1)
        add(boundedmin(g[:,:,k],(1,2)),f'projection_{offset}_min_SAT_m')
        add(boundedmin(np.where(el[:,:,k],g[:,:,k],np.inf),(1,2)),f'projection_{offset}_eligible_min_SAT_m')
        add(np.where(contact.any(1),contact.argmax(1)/40,1),f'projection_{offset}_first_contact_time_over4')
        add(contact.mean(1),f'projection_{offset}_contact_fraction')
    # Risk identity = minimum eligible margin; if absent, minimum any valid margin.
    eg=np.where(el,g,np.inf);has_el=np.isfinite(eg).any((1,2,3));risk=np.where(has_el[:,None,None,None],eg,g)
    ii=np.argmin(risk.reshape(M,-1),axis=1) if N else np.zeros(M,int)
    bb,kk,nn=np.unravel_index(ii,(B,4,max(N,1)))
    agrid=np.stack(actors,1)
    selected=agrid[bb,kk,nn] if N else np.zeros((M,7))
    anyvalid=np.isfinite(risk).any((1,2,3));iiM=np.arange(M);e=ex[iiM,bb];uf=fw[iiM,bb];vl=lf[iiM,bb]
    relpos=selected[:,:2]-e[:,:2];relvel=selected[:,3:5]-e[:,3:5];dyaw=selected[:,2]-e[:,2]
    type_=a['types'][nn] if N else np.full(M,-1);source=a['sources'][nn] if N else np.full(M,'invalid')
    vals=[(relpos*uf).sum(1),(relpos*vl).sum(1),(relvel*uf).sum(1),(relvel*vl).sum(1),np.sin(dyaw),np.cos(dyaw),ah[iiM,bb,kk,nn]if N else np.zeros(M),bh[iiM,bb,kk,nn]if N else np.zeros(M),type_==1,type_==2,type_==3,type_>=4,source=='physical',source=='frozen_factual',source=='current_CV_unsupported',bb/40,kk/3,np.linalg.norm(e[:,3:5],axis=-1),np.linalg.norm(selected[:,3:5],axis=-1),a['current_known'][nn]if N else np.zeros(M)]
    bn=['longitudinal_m','lateral_m','relative_longitudinal_mps','relative_lateral_mps','heading_sin','heading_cos','ahead','behind','vehicle','pedestrian','bicycle','other_type','physical','factual','currentCV','base_time_over4','projection_time_over0p9','ego_speed_mps','actor_speed_mps','current_known']
    for v,n in zip(vals,bn):add(np.where(anyvalid,v,0),'risk_entity_'+n)
    av=a['actor_valid'];den=max(int(av.sum()),1)
    support=[np.log1p(av.sum())]+[float(((a['sources']==q)&av).sum())/den for q in ['physical','frozen_factual','current_CV_unsupported']]+[float((a['current_known']&av).sum())/den,float(((a['types']<1)&av).sum())/den,known.mean(1),(multi&known).mean(1),(nondrive&known).mean(1),(inter&known).mean(1),speed.mean(1),speed.max(1),speed[:,-1],np.linalg.norm(ex[:,:B,5:7],axis=-1).max(1),np.abs(np.diff(np.unwrap(yaw,axis=1),axis=1)).sum(1),np.broadcast_to(allok,(M,*allok.shape)).mean((1,2,3))if N else 0]
    sn=['log1p_actor_count','physical_fraction','factual_fraction','currentCV_fraction','current_known_fraction','unknown_type_fraction','map_known_fraction','multiple_lanes_fraction','nondrive_fraction','intersection_fraction','ego_speed_mean_mps','ego_speed_max_mps','ego_speed_last_mps','ego_accel_max_mps2','heading_total_variation_rad','interpolation_support_fraction']
    for v,n in zip(support,sn):add(v,'support_'+n)
    actor_speed=np.linalg.norm(pred[:B,:,3:5],axis=-1)
    cond=[actor_speed<.05,np.broadcast_to(a['types']==1,(B,N)),np.broadcast_to(np.isin(a['types'],[2,3]),(B,N)),np.broadcast_to(a['types']>=4,(B,N)),ah[:,:,0],bh[:,:,0]]
    for mask,n in zip(cond,['stationary','vehicle','vulnerable','other_type','ahead','behind']):
        mask=np.broadcast_to(mask,(M,B,N))&pvalid[None,:B]
        add(boundedmin(np.where(mask,g[:,:,0],np.inf),(1,2)),'zero_projection_'+n+'_min_SAT_m')
        add(mask.any((1,2)),'zero_projection_'+n+'_supported')
    features=np.stack(f,-1).astype(np.float32);assert features.shape==(M,64) and np.isfinite(features).all()
    return np.where(valid[:,None],features,0),names,dict(risk_actor_index=np.where(anyvalid,nn,-1),risk_base_index=bb,risk_projection_index=kk,risk_used_eligible=has_el,actor_interpolation_valid=pvalid,map_known=known)

def current_areas(current,ex,map_root):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    from current_map_export import footprint
    import shapely
    assert current['latest_source_timestamp_us']<=current['timestamp_us']
    vehicle=get_pacifica_parameters();origin=np.asarray(current['rear_axle_pose']);c,s=np.cos(origin[2]),np.sin(origin[2]);p=ex[:,:,:3].copy();p[:,:,:2]=p[:,:,:2]@np.array([[c,s],[-s,c]])+origin[:2];p[:,:,2]+=origin[2]
    ego=EgoState.build_from_rear_axle(StateSE2(*origin),StateVector2D(*current['velocity']),StateVector2D(*current['acceleration']),0.,TimePoint(current['timestamp_us']),vehicle)
    coords=footprint(p,vehicle.length,vehicle.width,vehicle.rear_axle_to_center);shape=p.shape[:2];area={n:np.zeros(shape,bool)for n in ['known','multiple','nondrive','intersection']};info={}
    try:
        name='us-nv-las-vegas-strip'if current['map_name']=='las_vegas'else current['map_name'];api=get_maps_api(str(map_root),'nuplan-maps-v1.0',name);native=PDMDrivableMap.from_simulation(api,ego,100.)
        corner_in=np.zeros((*shape,4),bool);lane_count=np.zeros(shape,int);single_lane=np.zeros(shape,bool)
        for layer,poly in zip(native._map_types,native._geometries):
            inside=shapely.contains_xy(poly,coords[...,0],coords[...,1])[...,:4]
            if layer.name in ['ROADBLOCK','INTERSECTION','DRIVABLE_AREA','CARPARK_AREA']:corner_in|=inside
            if layer.name in ['LANE','LANE_CONNECTOR']:lane_count+=inside.any(-1);single_lane|=inside.all(-1)
            if layer.name=='INTERSECTION':area['intersection']|=shapely.contains_xy(poly,p[...,0],p[...,1])
        center=np.array([ego.center.x,ego.center.y]);area['known']=(np.linalg.norm(coords-center,axis=-1).max(-1)<=100)
        area['multiple']=(lane_count>1)&~single_lane;area['nondrive']=~corner_in.all(-1)
        info=dict(query_radius_m=100,polygon_count=len(native._geometries),route_required=False)
    except Exception as e:info=dict(error=repr(e),map_unknown=True)
    return area,info
