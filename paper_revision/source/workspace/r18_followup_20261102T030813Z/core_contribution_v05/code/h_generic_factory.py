"""R18 historical-platform Generic: fixed Frenet 0/+1m/-1m proposals.

Reuses the existing quintic Frenet formula and historical candidate +/-1m offsets. Historical
quintic initial curvature, time law, geometry, kinematic limits and no-stop
policy remain. No Supply-v2 physical gate, learned readout or fallback imported.
"""
import numpy as np
from shapely.geometry import Point,LineString
import revision_v5_twentyeighth_map_supply as historical

class GenericSupply(historical.MapSupply):
    def __init__(self,mp,cmap=None,module=None,offsets=(0.,1.,-1.)):
        super().__init__(mp,cmap);self.module=module or historical;self.offsets=offsets
    def generate(self,h):
        m=self.module;cfg=m.CONFIG;lanes,rec=self.targets(h);out=[]
        if not lanes:return out,rec
        lane=lanes[0];base=lane.baseline_path
        origin=np.r_[h['ego2global_translation'][:2],h['body_pose_math'][2]];r=m.rotation(origin[2]);s0=base.linestring.project(Point(origin[:2]));v0=float(h['ego_dynamic_state'][0]);a0=float(h['ego_dynamic_state'][2]);k0=float(h['omega']/v0)if abs(v0)>.5 else 0.
        near=base.get_nearest_pose_from_position(Point(origin[:2]));normal=np.array([-np.sin(near.heading),np.cos(near.heading)])
        width=float(np.clip(lane.polygon.intersection(LineString([np.array([near.x,near.y])-normal*8,np.array([near.x,near.y])+normal*8])).length,2.5,4.5))
        for slot,join in enumerate(cfg['join_lengths_m']):
            offset=self.offsets[slot%len(self.offsets)]
            detail=dict(join_length_m=join,target_lane=lane.row['key'],reference_lane=lane.row['key'],admitted_geometry=False,supply_source='historical_platform_Generic_Frenet',slot=slot,frenet_offset_m=offset,offset_units='meters_from_historical_group_cache',lane_width_m=width,geometry_version='historical_R18',speed_cap_mps=cfg['speed_max'],curvature_limit_per_m=cfg['curvature_limit_per_m'])
            if s0+join+1>base.length:
                out.append((None,detail|dict(reject='insufficient_current_lane_arc'),None));continue
            target=base.linestring.interpolate(s0+join);pose=base.get_nearest_pose_from_position(target)
            target_pose=np.array([pose.x-offset*np.sin(pose.heading),pose.y+offset*np.cos(pose.heading),pose.heading])
            legal=[ln.row['key']for ln in lanes if ln.polygon.covers(Point(target_pose[:2]))and np.cos(target_pose[2]-ln.baseline_path.get_nearest_pose_from_position(Point(target_pose[:2])).heading)>.5]
            detail.update(target_global=target_pose.tolist(),legal_target_lanes=legal)
            if not legal:
                out.append((None,detail|dict(reject='target_not_currently_legal_lane'),None));continue
            rel=(target_pose[:2]-origin[:2])@r;start_k=k0 if abs(v0)>.5 else float(2*rel[1]/max(rel@rel,1.));detail['reference_initial_curvature']=start_k
            k1=float(base.get_curvature_at_arc_length(s0+join));den=1-offset*k1
            if den<=.25:
                out.append((None,detail|dict(reject='singular_frenet_offset'),None));continue
            q,k=m.quintic(origin,target_pose,join,start_k,k1/den)
            tail=[];tk=[];singular=False
            for s in np.arange(s0+join+.1,min(base.length,s0+join+30),.1):
                pp=base.linestring.interpolate(s);po=base.get_nearest_pose_from_position(pp);kk=float(base.get_curvature_at_arc_length(s));den=1-offset*kk
                if den<=.25:singular=True;break
                tail.append([po.x-offset*np.sin(po.heading),po.y+offset*np.cos(po.heading),po.heading]);tk.append(kk/den)
            if singular:
                out.append((None,detail|dict(reject='singular_frenet_offset_tail'),None));continue
            path=np.r_[q,np.asarray(tail).reshape(-1,3)];kk=np.r_[k,tk];detail['max_curvature']=float(abs(kk).max())
            if abs(kk).max()>cfg['curvature_limit_per_m']:
                out.append((None,detail|dict(reject='curvature'),path));continue
            timed,v,a=m.time_parameterize(path,kk,v0,a0);local=timed.copy();local[:,:2]=(local[:,:2]-origin[:2])@r;local[:,2]=np.arctan2(np.sin(local[:,2]-origin[2]),np.cos(local[:,2]-origin[2]));ca=local[10::10].astype(np.float32);dist,token=m.separation(local,h)
            full=np.r_[[[0.,0.,0.]],ca];speed=np.linalg.norm(np.diff(full[:,:2],axis=0),axis=1)/.5;acc=np.diff(np.r_[max(0,v0),speed])/.5;yr=np.diff(np.unwrap(full[:,2]))/.5
            kin=bool(max(acc)<=3.15 and min(acc)>=-6.15 and max(abs(yr))<=.55 and max(speed*abs(yr))<=4.2)
            detail.update(current_box_minSAT_m=float(dist),nearest_static_actor=token,kinematically_admissible=kin,initial_v=v0,initial_a=a0,terminal_local=ca[-1].tolist(),candidate=ca.tolist(),max_yaw_rate=float(abs(yr).max()),max_acceleration=float(max(acc)),min_acceleration=float(min(acc)),admitted_geometry=bool(kin and dist>=cfg['current_box_clearance_m']))
            if not detail['admitted_geometry']:detail['reject']='kinematics'if not kin else'current_stationary_box'
            out.append((ca if detail['admitted_geometry']else None,detail,local))
        return out,rec
