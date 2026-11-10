"""Supply-v2 common geometric proposal factory, with at most 3+2 attempts.

Map allocates attempts across currently connected legal lanes. Generic samples
center / +lane-width / -lane-width offsets about the nearest reference lane.
Both use the same quintic helper, horizons, speed, topology legality, vehicle
limits, static boxes, and downstream controller-prefix checker. No CW head.
"""
import math
import numpy as np
from shapely.geometry import Point
from revision_v5_twentyeighth_map_supply import MapSupply,quintic,separation
from cw_followup_physics import rotation

CONFIG=dict(version='Supply_v2_draft02',max_extra_attempts=5,normal_attempts=3,slow_attempts=2,
    normal_joins_m=[8.,10.,12.],slow_profiles=[dict(join_m=8.,speed_mps=.8),dict(join_m=6.,speed_mps=.8)],
    normal_speed_mps=3.5,normal_curvature_cap=.30,slow_curvature_cap=.50,
    measured_steer_capacity_fraction=.8,accel_max=1.5,jerk_max=2.,yaw_rate_max=.5,lat_accel_max=3.,
    clearance_m=.1,positive_progress_m=2.,trigger_max_speed_mps=2.,terminal_heading_cos=.95,
    generic_offsets_lane_widths=[0.,1.,-1.],generic_source='nearest legal reference lane Frenet offsets, no target-lane allocation',
    map_allocation='uncovered legal lanes first, stable near-distance/lane-id order, cyclic revisit')

def time_path(path,curvature,v0,a0,speed_cap):
    arc=np.r_[0,np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).cumsum()]
    keep=np.r_[True,np.diff(arc)>1e-5];path,curvature,arc=path[keep],curvature[keep],arc[keep]
    v=max(0.,float(v0));a=float(np.clip(a0,-3.,3.));s=0.;ss=[0.]
    k=float(abs(curvature).max(initial=0));cap=min(speed_cap,.9*CONFIG['yaw_rate_max']/max(k,1e-5),math.sqrt(CONFIG['lat_accel_max']/max(k,1e-5)))
    for _ in range(80):
        target=min(cap,math.sqrt(max(0.,3*(arc[-1]-s))));desired=float(np.clip(2*(target-v),-3,CONFIG['accel_max']));a+=np.clip(desired-a,-CONFIG['jerk_max']*.05,CONFIG['jerk_max']*.05)
        nv=max(0.,v+a*.05);s=min(arc[-1],s+(v+nv)*.025);v=nv;ss.append(s)
    return np.stack([np.interp(ss,arc,path[:,0]),np.interp(ss,arc,path[:,1]),np.interp(ss,arc,np.unwrap(path[:,2]))],1)

class Factory(MapSupply):
    def __init__(self,mp,cmap,physics):
        super().__init__(mp,cmap);self.physics=physics
    def terminal(self,pose,lanes):
        matches=[]
        for lane in lanes:
            q=lane.baseline_path.get_nearest_pose_from_position(Point(pose[:2]))
            if lane.polygon.covers(Point(pose[:2])) and np.cos(pose[2]-q.heading)>.5:matches.append(lane.row['key'])
        return matches
    def proposal(self,h,lane,join,offset,speed_cap,slow,attempt):
        base=lane.baseline_path;origin=np.r_[h['ego2global_translation'][:2],h['body_pose_math'][2]];r=rotation(origin[2]);s0=base.linestring.project(Point(origin[:2]));v0=float(h['ego_dynamic_state'][0]);a0=float(h['ego_dynamic_state'][2])
        detail=dict(attempt=attempt,join_length_m=join,reference_lane=lane.row['key'],frenet_offset_m=offset,speed_cap_mps=speed_cap,slow=slow,admitted_geometry=False)
        if s0+join+1>base.length:return None,detail|dict(reject='insufficient_current_lane_arc'),None
        pt=base.linestring.interpolate(s0+join);q=base.get_nearest_pose_from_position(pt);target=np.array([q.x-offset*np.sin(q.heading),q.y+offset*np.cos(q.heading),q.heading])
        rel=(target[:2]-origin[:2])@r
        measured=h['controller_state']['measured_front_steer_rad']
        start_k=math.tan(measured)/self.physics['wheelbase_m']
        # C2 initial curvature is measured steering, including standstill.
        # A nominal geometric turn curvature is not an executed initial state.
        target_k=float(base.get_curvature_at_arc_length(s0+join));denominator=1-offset*target_k
        if denominator<=.25:return None,detail|dict(reject='singular_frenet_offset',frenet_denominator=denominator),None
        target_k=target_k/denominator
        path,k=quintic(origin,target,join,start_k,target_k)
        tail=[];tk=[]
        for s in np.arange(s0+join+.1,min(base.length,s0+join+30),.1):
            po=base.get_nearest_pose_from_position(base.linestring.interpolate(s));kk=float(base.get_curvature_at_arc_length(s));denominator=1-offset*kk
            if denominator<=.25:return None,detail|dict(reject='singular_frenet_offset_tail',frenet_denominator=denominator),None
            tail.append([po.x-offset*np.sin(po.heading),po.y+offset*np.cos(po.heading),po.heading]);tk.append(kk/denominator)
        path=np.r_[path,np.asarray(tail).reshape(-1,3)];k=np.r_[k,tk]
        peak_k=float(abs(k).max());planned_speed=min(speed_cap,.9*CONFIG['yaw_rate_max']/max(peak_k,1e-5),math.sqrt(CONFIG['lat_accel_max']/max(peak_k,1e-5)))
        curve=np.asarray(self.physics['steering_curve']);scale=float(np.interp(max(abs(v0),planned_speed)*3.6,curve[:,0],curve[:,1]));inner=math.radians(max(w['max_steer_angle_deg'] for w in self.physics['wheels']))*scale
        physical=CONFIG['measured_steer_capacity_fraction']*math.tan(inner)/(self.physics['wheelbase_m']+self.physics['front_track_m']/2*abs(math.tan(inner)))
        cap=min(physical,CONFIG['slow_curvature_cap'] if slow else CONFIG['normal_curvature_cap']);detail.update(initial_measured_curvature=start_k,max_curvature=float(abs(k).max()),curvature_cap=cap)
        if abs(k).max()>cap:return None,detail|dict(reject='vehicle_curvature'),None
        timed=time_path(path,k,v0,a0,speed_cap);local=timed.copy();local[:,:2]=(local[:,:2]-origin[:2])@r;local[:,2]=np.arctan2(np.sin(local[:,2]-origin[2]),np.cos(local[:,2]-origin[2]))
        ca=local[10::10].astype(np.float32);sep,token=separation(local,h);full=np.r_[[[0.,0.,0.]],ca];speed=np.linalg.norm(np.diff(full[:,:2],axis=0),axis=1)/.5;acc=np.diff(np.r_[max(v0,0.),speed])/.5;yr=np.diff(np.unwrap(full[:,2]))/.5
        kinetic=bool(acc.max()<=3.15 and acc.min()>=-6.15 and abs(yr).max()<=.55 and (speed*abs(yr)).max()<=4.2)
        lanes,top=self.targets(h);legal_target=self.terminal(target,lanes)
        detail.update(geometric_static_minSAT_m=float(sep),static_actor=token,kinematically_admissible=kinetic,legal_target_lanes=legal_target,terminal_local=ca[-1].tolist(),candidate=ca.tolist(),target_global=target.tolist())
        reason=None if kinetic and sep>=CONFIG['clearance_m'] and legal_target else ('kinematics' if not kinetic else 'geometry_static_bbox' if sep<CONFIG['clearance_m'] else 'target_not_currently_legal_lane')
        detail.update(admitted_geometry=reason is None,reject=reason)
        return ca if reason is None else None,detail,local
    def generate_stage(self,h,mode,covered=(),slow=False):
        lanes,top=self.targets(h)
        if not lanes:return [],top
        ordered=sorted(enumerate(lanes),key=lambda x:(x[1].row['key'] in covered,x[0]))
        out=[];n=CONFIG['slow_attempts'] if slow else CONFIG['normal_attempts']
        nearest=lanes[0]
        # Actual polygon width at the nearest baseline point; bounded to sane
        # lane width range only to avoid degenerate map export measurements.
        q=nearest.baseline_path.get_nearest_pose_from_position(Point(h['ego2global_translation'][:2]));normal=np.array([-np.sin(q.heading),np.cos(q.heading)])
        from shapely.geometry import LineString
        width=float(nearest.polygon.intersection(LineString([np.array([q.x,q.y])-normal*8,np.array([q.x,q.y])+normal*8])).length)
        width=float(np.clip(width,2.5,4.5))
        for j in range(n):
            lane=ordered[j%len(ordered)][1] if mode=='Map' else nearest
            offset=0. if mode=='Map' or slow else CONFIG['generic_offsets_lane_widths'][j]*width
            profile=CONFIG['slow_profiles'][j] if slow else dict(join_m=CONFIG['normal_joins_m'][j],speed_mps=CONFIG['normal_speed_mps'])
            ca,d,path=self.proposal(h,lane,profile['join_m'],offset,profile['speed_mps'],slow,j+(3 if slow else 0));d.update(mode=mode,legal_lanes_total=len(lanes),generic_lane_width_m=width if mode=='Generic' else None);out.append((ca,d,path))
        return out,top
