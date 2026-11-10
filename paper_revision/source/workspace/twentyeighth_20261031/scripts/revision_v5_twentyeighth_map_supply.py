"""R18 spatial supply: current-state C2 merge into a connected driving lane.

No route/actor identity cases, scenario metadata, future observations, or labels.
Parking entry is distinct from a Driving-to-Driving marked lane change. The
marking and source lane type are recorded, including Solid parking boundaries.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
from shapely.geometry import Point

CONFIG=dict(version='MapSupply_v04',join_lengths_m=[8.,10.,12.],max_attempts=3,
    curvature_limit_per_m=.30,accel_max=1.5,jerk_max=2.,speed_max=3.5,
    yaw_rate_max=.5,lat_accel_max=3.,stationary_actor_speed=.2,
    current_box_clearance_m=.10,positive_progress_m=2.,slow_speed_mps=2.,
    stalled_speed_mps=.5,stalled_distance_m=.25,stalled_history_s=1.,
    max_entry_distance_m=6.,history_EC_is_score_not_hard_gate=True,P0=False)

def rotation(h):
    c,s=np.cos(h),np.sin(h);return np.array([[c,-s],[s,c]])

def separation(path,h):
    """Actual Lincoln box vs currently stationary observations, sampled sweep."""
    boxes=np.asarray(h['anns']['gt_boxes'],float).reshape(-1,7)
    vel=np.asarray(h['anns']['gt_velocity_3d'],float).reshape(-1,3)
    ids=np.flatnonzero(np.linalg.norm(vel[:,:2],axis=1)<=CONFIG['stationary_actor_speed'])
    if not len(ids):return np.inf,None
    g=h['rear_geometry'];extent=np.asarray(g['box_extent'][:2]);off=np.array([g['box_offset'][0],-g['box_offset'][1]])-g['rear_offset_math']
    path=np.asarray(path);bb=boxes[ids];R=np.stack([rotation(y)for y in path[:,2]]);RB=np.stack([rotation(y)for y in bb[:,6]])
    center=path[:,:2]+R@off
    axes=np.concatenate([np.broadcast_to(R.transpose(0,2,1)[:,None],(len(path),len(ids),2,2)),np.broadcast_to(RB.transpose(0,2,1)[None],(len(path),len(ids),2,2))],axis=2)
    distance=abs(np.einsum('saki,sai->sak',axes,bb[None,:,:2]-center[:,None]))
    radii=abs(np.einsum('saki,sij->sakj',axes,R))@extent+np.einsum('sakj,aj->sak',abs(np.einsum('saki,aij->sakj',axes,RB)),bb[:,3:5]/2)
    sep=(distance-radii).max(-1);i,j=np.unravel_index(sep.argmin(),sep.shape)
    return float(sep[i,j]),str(h['anns']['track_tokens'][ids[j]])

def local_execution(ex,h):
    o=np.r_[h['ego2global_translation'][:2],h['body_pose_math'][2]]
    p=np.array(ex[...,:3],copy=True);p[...,:2]=(p[...,:2]-o[:2])@rotation(o[2]);p[...,2]=np.arctan2(np.sin(p[...,2]-o[2]),np.cos(p[...,2]-o[2]));return p

def coverage(a,native,pred,h,allowed_lanes):
    paths=local_execution(native,h);safe=[];sat=[]
    for i in range(len(a['valid'])):
        d,t=separation(paths[i],h)if a['valid'][i]else(-np.inf,None);sat.append(d)
        terminal_lane=any(l.polygon.covers(Point(native[i,-1,:2]))for l in allowed_lanes)
        safe.append(bool(a['valid'][i]and pred['eligible'][i]and d>=CONFIG['current_box_clearance_m']and paths[i,-1,0]>=CONFIG['positive_progress_m']and terminal_lane))
    return np.asarray(safe),np.asarray(sat)

def quintic(start,target,length,k0=0.,k1=0.):
    p0=np.asarray(start[:2]);p1=np.asarray(target[:2]);v0=length*np.array([np.cos(start[2]),np.sin(start[2])]);v1=length*np.array([np.cos(target[2]),np.sin(target[2])]);a0=k0*length**2*np.array([-np.sin(start[2]),np.cos(start[2])]);a1=k1*length**2*np.array([-np.sin(target[2]),np.cos(target[2])])
    co=np.zeros((6,2));co[:3]=[p0,v0,a0/2]
    co[3:]=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]]),np.array([p1-co[:3].sum(0),v1-co[1]-2*co[2],a1-2*co[2]]))
    t=np.linspace(0,1,max(81,int(length/.05)+1));xy=np.stack([t**i for i in range(6)],1)@co
    v=np.stack([i*t**(i-1)for i in range(1,6)],1)@co[1:];a=np.stack([i*(i-1)*t**(i-2)for i in range(2,6)],1)@co[2:]
    hd=np.arctan2(v[:,1],v[:,0]);k=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(np.linalg.norm(v,axis=1)**3,1e-8)
    return np.c_[xy,hd],k

def time_parameterize(path,k,v0,a0):
    arc=np.r_[0,np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).cumsum()]
    use=np.r_[True,np.diff(arc)>1e-5];path,k,arc=path[use],k[use],arc[use]
    vel=max(0.,float(v0));acc=float(np.clip(a0,-3.,3.));s=0.;ss=[s];vv=[vel];aa=[acc]
    # The entire proposed merge is known now. Reserve curvature speed before
    # entering the turn; a one-second local cap with jerk lag overshot yaw rate.
    kap=float(np.max(abs(k),initial=0));path_cap=min(CONFIG['speed_max'],.9*CONFIG['yaw_rate_max']/max(kap,1e-5),np.sqrt(CONFIG['lat_accel_max']/max(kap,1e-5)))
    for _ in range(80):
        cap=min(path_cap,np.sqrt(max(0,2*1.5*(arc[-1]-s))))
        goal=float(np.clip((cap-vel)*2,-3.,CONFIG['accel_max']));acc+=np.clip(goal-acc,-CONFIG['jerk_max']*.05,CONFIG['jerk_max']*.05);new=max(0.,vel+acc*.05);s=min(arc[-1],s+(vel+new)*.025);vel=new;ss.append(s);vv.append(vel);aa.append(acc)
    poses=np.stack([np.interp(ss,arc,path[:,0]),np.interp(ss,arc,path[:,1]),np.interp(ss,arc,np.unwrap(path[:,2]))],1)
    return poses,np.asarray(vv),np.asarray(aa)

class MapSupply:
    def __init__(self,mp,cmap=None):
        self.mp=mp
        if cmap is None:
            import carla
            B=Path('<CW_WORKSPACE>/twentyeighth_20261031')
            item=next(x for x in json.load(open(B/'research/COVERAGE_MAP_ASSETS.json'))if x['town']==mp.map_name.split('/')[-1]);raw=Path(item['opendrive'][0]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['opendrive'][0]['sha256'];cmap=carla.Map(mp.map_name,raw.decode())
        self.cmap=cmap

    def targets(self,h):
        import carla
        from revision_v5_twentyeighth_carla_view import key
        p=h['body_pose_math'];loc=carla.Location(x=p[0],y=-p[1],z=h['ego2global_translation'][2]);src=self.cmap.get_waypoint(loc,True,carla.LaneType.Any);near=self.cmap.get_waypoint(loc,True,carla.LaneType.Driving)
        if src is None or near is None:return [],dict(reason='no_current_waypoint')
        kind=str(src.lane_type);allowed=[]
        if kind=='Parking':
            # Parking exit, not permission to cross a solid Driving lane line.
            for w in [src.get_left_lane(),src.get_right_lane()]:
                if w is not None and w.lane_type==carla.LaneType.Driving and w.lane_id*src.lane_id>0:allowed.append(w)
        elif kind=='Driving':
            allowed.append(src)
            for side,flag in [('left',carla.LaneChange.Left),('right',carla.LaneChange.Right)]:
                w=getattr(src,'get_'+side+'_lane')()
                if getattr(src,side+'_lane_marking').lane_change&flag and w is not None and w.lane_type==carla.LaneType.Driving and w.lane_id*src.lane_id>0:allowed.append(w)
        receipt=dict(source_type=kind,source_key=key(src),left_marking=str(src.left_lane_marking.type),right_marking=str(src.right_lane_marking.type),lane_change=str(src.lane_change),entry_semantics='adjacent same-direction Parking-to-Driving'if kind=='Parking'else'marked Driving connectivity',source_is_scenario_metadata=False)
        lanes=[];o=np.array(h['ego2global_translation'][:2]);yaw=h['body_pose_math'][2]
        for w in allowed:
            lane=self.mp.lanes.get(key(w))
            if lane is None or lane.parent.id not in h['roadblock_ids']:continue
            q=lane.baseline_path.get_nearest_pose_from_position(type('P',(),dict(x=o[0],y=o[1]))())
            if np.linalg.norm(np.array([q.x,q.y])-o)>CONFIG['max_entry_distance_m']or np.cos(q.heading-yaw)<.5:continue
            lanes.append(lane)
        lanes.sort(key=lambda l:(l.baseline_path.linestring.distance(Point(o)),l.id))
        receipt['target_lanes']=[l.row['key']for l in lanes];return lanes,receipt

    def generate(self,h):
        lanes,rec=self.targets(h);out=[]
        if not lanes:return out,rec
        lane=lanes[0];base=lane.baseline_path;origin=np.r_[h['ego2global_translation'][:2],h['body_pose_math'][2]];R=rotation(origin[2]);s0=base.linestring.project(Point(origin[:2]));v0=float(h['ego_dynamic_state'][0]);a0=float(h['ego_dynamic_state'][2]);k0=float(h['omega']/v0)if abs(v0)>.5 else 0.
        for join in CONFIG['join_lengths_m']:
            detail=dict(join_length_m=join,target_lane=lane.row['key'],admitted_geometry=False)
            if s0+join+1>base.length:detail['reject']='insufficient_current_lane_arc';out.append((None,detail,None));continue
            target=base.linestring.interpolate(s0+join);pose=base.get_nearest_pose_from_position(target)
            # At rest curvature is unobservable from yaw_rate/speed. A geometric
            # entry curvature avoids postponing the turn until the front box.
            # Native execution still starts with the measured actual state.
            rel=(np.array([pose.x,pose.y])-origin[:2])@R
            start_k=k0 if abs(v0)>.5 else float(2*rel[1]/max(rel@rel,1.))
            detail['reference_initial_curvature']=start_k
            q,k=quintic(origin,[pose.x,pose.y,pose.heading],join,start_k,base.get_curvature_at_arc_length(s0+join))
            # Continue the actual lane geometry; no straight extension beyond map.
            arc=np.arange(s0+join+.1,min(base.length,s0+join+30),.1);tail=[];tk=[]
            for s in arc:
                pp=base.linestring.interpolate(s);po=base.get_nearest_pose_from_position(pp);tail.append([po.x,po.y,po.heading]);tk.append(base.get_curvature_at_arc_length(s))
            path=np.r_[q,np.asarray(tail).reshape(-1,3)];kk=np.r_[k,tk];detail['max_curvature']=float(abs(kk).max())
            if abs(kk).max()>CONFIG['curvature_limit_per_m']:detail['reject']='curvature';out.append((None,detail,path));continue
            timed,v,a=time_parameterize(path,kk,v0,a0);local=timed.copy();local[:,:2]=(local[:,:2]-origin[:2])@R;local[:,2]=np.arctan2(np.sin(local[:,2]-origin[2]),np.cos(local[:,2]-origin[2]));ca=local[10::10].astype(np.float32);dist,token=separation(local,h)
            full=np.r_[[[0.,0.,0.]],ca];speed=np.linalg.norm(np.diff(full[:,:2],axis=0),axis=1)/.5;acc=np.diff(np.r_[max(0,v0),speed])/.5;yr=np.diff(np.unwrap(full[:,2]))/.5
            kin=bool(max(acc)<=3.15 and min(acc)>=-6.15 and max(abs(yr))<=.55 and max(speed*abs(yr))<=4.2)
            detail.update(current_box_minSAT_m=float(dist),nearest_static_actor=token,kinematically_admissible=kin,initial_v=v0,initial_a=a0,terminal_local=ca[-1].tolist(),max_yaw_rate=float(abs(yr).max()),max_acceleration=float(max(acc)),min_acceleration=float(min(acc)),admitted_geometry=bool(kin and dist>=CONFIG['current_box_clearance_m']))
            if not detail['admitted_geometry']:detail['reject']='kinematics'if not kin else'current_stationary_box'
            out.append((ca if detail['admitted_geometry']else None,detail,local))
        return out,rec
