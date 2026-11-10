"""Semantics-preserving prepared-geometry optimization of B20 signal features.

No labels, metric cache, future traffic lights or counterfactual successor logs.
Red/green channels are CURRENT-state persistence hypotheses, not future truth.
Forward stopping margins use a straight-heading rigid footprint and immediate
constant deceleration; they are diagnostics, not dynamic feasibility guarantees.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Point
from current_map_export import footprint

NAMES=['executed_end_x_over100','executed_end_y_over100','executed_end_yaw_sin','executed_end_yaw_cos',
       'executed_end_speed_over20','executed_end_long_accel_over10','executed_arc_over100','max_deceleration_over10',
       'first_stop_time_over4','terminal_stop_d_a3_over100','terminal_stop_d_a6_over100','initial_speed_over20',
       'submitted_end_x_over100','submitted_end_y_over100','submitted_end_yaw_sin','submitted_end_yaw_cos',
       'submitted_arc_over100','execution_minus_submission_x_over10','execution_minus_submission_y_over10','execution_minus_submission_yaw_over_pi',
       'map_known','log1p_observed_signal_count','log1p_matched_signal_count','log1p_current_red_count','log1p_current_green_count','log1p_unobserved_regulated_count',
       'red_overlap_any','red_overlap_initial','red_first_overlap_over4','red_overlap_fraction','red_min_clearance_over50','red_end_clearance_over50',
       'green_overlap_any','green_first_overlap_over4','unknown_control_overlap_any','unknown_control_first_overlap_over4','log1p_missing_signal_geometry',
       'initial_forward_red_distance_over100','initial_forward_red_found','terminal_forward_red_distance_over100','terminal_forward_red_found',
       'initial_required_deceleration_over10','initial_stop_margin_a3_over100','initial_stop_margin_a6_over100',
       'terminal_stop_margin_a3_over100','terminal_stop_margin_a6_over100']


class SignalGeometry:
    def __init__(self,map_root):
        self.map_root=str(map_root);self.apis={};self.objects={}

    def get(self,current,execution):
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        name=current['map_name'];name='us-nv-las-vegas-strip'if name=='las_vegas'else name
        if name not in self.apis:self.apis[name]=get_maps_api(self.map_root,'nuplan-maps-v1.0',name)
        api=self.apis[name];origin=np.asarray(current['rear_axle_pose']);status={str(t):bool(red)for t,red in current['traffic_lights']}
        radius=max(100.,float(np.linalg.norm(execution[:,:,:2],axis=-1).max())+105.)
        near=api.get_proximal_map_objects(Point2D(*origin[:2]),radius,[L.LANE_CONNECTOR])[L.LANE_CONNECTOR]
        edges={str(e.id):e for e in near}
        missing=[]
        for token in status:
            key=(name,token)
            if key not in self.objects:self.objects[key]=api.get_map_object(token,L.LANE_CONNECTOR)
            if self.objects[key]is not None:edges[token]=self.objects[key]
            else:missing.append(token)
        records=[];groups={'red':[],'green':[],'unknown':[]}
        for token,e in sorted(edges.items()):
            light=getattr(e,'has_traffic_lights',False);regulated=bool(light()if callable(light)else light)
            if token not in status and not regulated:continue
            state='red'if status.get(token)is True else('green'if token in status else'unknown')
            poly=e.polygon
            if not poly.is_valid:poly=shapely.make_valid(poly)
            if poly.is_empty:missing.append(token);continue
            groups[state].append(poly)
            records.append(dict(token=token,state=state,regulated=regulated,roadblock=str(e.get_roadblock_id()),wkb_hex=shapely.to_wkb(poly,hex=True)))
        unions={k:shapely.union_all(v)if v else shapely.Polygon()for k,v in groups.items()}
        receipt=dict(current_timestamp_us=current['timestamp_us'],latest_source_timestamp_us=current['latest_source_timestamp_us'],map_name=name,query_radius_m=radius,known_signal_count=len(status),matched_signal_count=sum(t in edges and edges[t]is not None for t in status),missing_signal_geometry=missing,records=records,future_fields_used=False,state_semantics='current only; red persistence is a hypothesis, unknown is neither green nor future red')
        return unions,receipt


def global_poses(local,current):
    origin=np.asarray(current['rear_axle_pose']);c,s=np.cos(origin[2]),np.sin(origin[2]);p=np.asarray(local,dtype=float).copy();p[...,:2]=p[...,:2]@np.array([[c,s],[-s,c]])+origin[:2];p[...,2]+=origin[2];return p


def polygons(poses,vehicle):
    return shapely.polygons(footprint(poses,*vehicle)[...,:4,:])


def forward_distance(poses,red,vehicle):
    """First intersection on 0.25m straight-heading samples, max100m.

    Not a physical stop line or curved-road reachable-set proof. No hit means
    unknown/outside probe support, encoded with found=False, not infinite safety.
    """
    if red.is_empty:return np.full(len(poses),100.),np.zeros(len(poses),bool)
    d=np.arange(401)*.25;p=np.repeat(poses[:,None,:],len(d),axis=1)
    p[...,:2]+=np.stack([np.cos(poses[:,2]),np.sin(poses[:,2])],-1)[:,None]*d[None,:,None]
    shapely.prepare(red);hits=shapely.intersects(red,polygons(p,vehicle));found=hits.any(1);distance=np.where(found,hits.argmax(1)*.25,100.)
    return distance,found


def features(current,assets,candidates,extractor):
    assert current['latest_source_timestamp_us']<=current['timestamp_us']
    ex=np.asarray(assets['execution_full_local'],float);valid=assets['candidate_valid'];m=len(valid)
    assert ex.shape==(m,41,11) and candidates.shape==(m,8,3)
    v=np.linalg.norm(ex[:,:,3:5],axis=-1);along=np.sum(ex[:,:,5:7]*np.stack([np.cos(ex[:,:,2]),np.sin(ex[:,:,2])],-1),-1)
    arc=np.linalg.norm(np.diff(ex[:,:,:2],axis=1),axis=-1).sum(1);stop=v<.3
    sub_arc=np.linalg.norm(np.diff(np.concatenate([np.zeros((m,1,2)),candidates[:,:,:2]],1),axis=1),axis=-1).sum(1)
    end=ex[:,-1];sub=candidates[:,-1];err=np.arctan2(np.sin(end[:,2]-sub[:,2]),np.cos(end[:,2]-sub[:,2]));init=np.linalg.norm(current['velocity'])
    vals=[end[:,0]/100,end[:,1]/100,np.sin(end[:,2]),np.cos(end[:,2]),v[:,-1]/20,along[:,-1]/10,arc/100,np.maximum(-along.min(1),0)/10,np.where(stop.any(1),stop.argmax(1)/40,1),v[:,-1]**2/6/100,v[:,-1]**2/12/100,np.full(m,init/20),sub[:,0]/100,sub[:,1]/100,np.sin(sub[:,2]),np.cos(sub[:,2]),sub_arc/100,(end[:,0]-sub[:,0])/10,(end[:,1]-sub[:,1])/10,err/np.pi]
    world=global_poses(ex[:,:,:3],current);vehicle=(float(assets['ego_box'][0]),float(assets['ego_box'][1]),float(assets['rear_axle_offset'].ravel()[0]))
    unions,receipt=extractor.get(current,ex);body=polygons(world,vehicle);groups={}
    for key,poly in unions.items():
        shapely.prepare(poly)
        hit=np.zeros((m,41),bool)if poly.is_empty else shapely.intersects(poly,body)
        distance=np.full((m,41),50.)if poly.is_empty else np.minimum(shapely.distance(body,poly),50.)
        groups[key]=dict(hit=hit,distance=distance)
    nr=lambda key:sum(r['state']==key for r in receipt['records'])
    vals.extend([np.ones(m),np.full(m,np.log1p(receipt['known_signal_count'])),np.full(m,np.log1p(receipt['matched_signal_count'])),np.full(m,np.log1p(nr('red'))),np.full(m,np.log1p(nr('green'))),np.full(m,np.log1p(nr('unknown')))])
    red=groups['red'];h=red['hit'];first=lambda hit:np.where(hit.any(1),hit.argmax(1)/40,1)
    vals.extend([h.any(1),h[:,0],first(h),h.mean(1),red['distance'].min(1)/50,red['distance'][:,-1]/50])
    for key in ['green','unknown']:vals.extend([groups[key]['hit'].any(1),first(groups[key]['hit'])])
    vals.append(np.full(m,np.log1p(len(receipt['missing_signal_geometry']))))
    d0,k0=forward_distance(world[:1,0],unions['red'],vehicle);de,ke=forward_distance(world[:,-1],unions['red'],vehicle);d0=np.full(m,d0[0]);k0=np.full(m,k0[0])
    vals.extend([d0/100,k0,de/100,ke,np.where(k0,np.minimum(init**2/(2*np.maximum(d0,.01)),20),0)/10,np.where(k0,d0-init**2/6,0)/100,np.where(k0,d0-init**2/12,0)/100,np.where(ke,de-v[:,-1]**2/6,0)/100,np.where(ke,de-v[:,-1]**2/12,0)/100])
    f=np.stack(vals,-1).astype(np.float32);assert f.shape==(m,len(NAMES)) and np.isfinite(f).all()
    return np.where(valid[:,None],f,0),dict(red_hit=h,green_hit=groups['green']['hit'],unknown_hit=groups['unknown']['hit'],initial_red_distance=d0,initial_red_found=k0,terminal_red_distance=de,terminal_red_found=ke,executed_terminal=end,submitted_terminal=sub),receipt
