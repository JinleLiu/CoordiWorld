"""Current CARLA observation -> right-handed rear-axle frame; no future actors."""
import math,json,hashlib
from pathlib import Path
import numpy as np
def key(w):return f'{w.road_id}:{w.section_id}:{w.lane_id}'
def pose(t):return np.array([t.location.x,-t.location.y,-math.radians(t.rotation.yaw)])
def rot(h):return np.array([[math.cos(h),-math.sin(h)],[math.sin(h),math.cos(h)]])
def world_vector(v):return np.array([v.x,-v.y])
def vehicle_geometry(hero):
 t=hero.get_transform();inv=np.asarray(t.get_inverse_matrix());w=hero.get_physics_control().wheels
 # Physics wheels are Unreal-world cm. LargeMap rebases this origin while
 # actor.get_transform exposes global map coordinates. Never conflate them.
 raw=np.array([[x.position.x,x.position.y,x.position.z]for x in w])/100.;local=np.array([(inv@np.r_[x,1])[:3]for x in raw]);origin_receipt={}
 if abs(local[:,:2]).max()>10:
  p=Path('<CW_WORKSPACE>/twentyeighth_20261031/bridge/calibration/geometry.json');cal=json.load(open(p))['records'];assert all(x['hero_type']==hero.type_id for x in cal)
  fixed=np.mean([x['wheels_body_m']for x in cal],axis=0);mat=np.asarray(t.get_matrix());expected=np.array([(mat@np.r_[x,1])[:3]for x in fixed]);shift=expected-raw;residual=shift-shift.mean(0)
  assert np.linalg.norm(residual[:,:2],axis=1).max()<.08,('wheel geometry changed',residual.tolist())
  local=fixed;origin_receipt=dict(rebased_world=True,inferred_physics_to_map_origin_m=shift.mean(0).tolist(),four_wheel_translation_residual_m=residual.tolist(),calibration_sha=hashlib.sha256(p.read_bytes()).hexdigest(),rear_geometry='two-placement same-blueprint local calibration; no guessed global map origin')
 order=np.argsort(local[:,0]);rear=local[order[:2]].mean(0);front=local[order[-2:]].mean(0)
 assert 1.5<front[0]-rear[0]<5 and abs(rear[1])<.3,(local.tolist(),rear.tolist())
 return dict(rear_offset_math=[float(rear[0]),float(-rear[1])],wheelbase=float(front[0]-rear[0]),wheels_body_m=local.tolist(),body_pose_math=pose(t).tolist(),box_extent=[hero.bounding_box.extent.x,hero.bounding_box.extent.y,hero.bounding_box.extent.z],box_offset=[hero.bounding_box.location.x,hero.bounding_box.location.y,hero.bounding_box.location.z],world_to_body_matrix=inv.tolist(),wheel_position_units='Unreal-world centimetres; account for large-map origin rebasing',**origin_receipt)
def export_map(cmap,path):
 path=Path(path)
 if path.exists():return json.loads(path.read_text())
 import xml.etree.ElementTree as ET
 od=cmap.to_opendrive();roads={}
 for rd in ET.fromstring(od).findall('road'):
  speeds=[]
  for sp in rd.findall('./type/speed'):
   try:v=float(sp.attrib['max']);v*= {'mph':.44704,'km/h':1/3.6,'m/s':1}.get(sp.attrib.get('unit','m/s'),1);speeds.append(v)
   except(ValueError,KeyError):pass
  roads[int(rd.attrib['id'])]=min(speeds)if speeds else None
 groups={}
 for w in cmap.generate_waypoints(1.0):groups.setdefault(key(w),[]).append(w)
 for a,b in cmap.get_topology():
  groups.setdefault(key(a),[]).append(a);groups.setdefault(key(b),[]).append(b)
 rows=[]
 for k,ww in sorted(groups.items()):
  ww=sorted(ww,key=lambda x:x.s,reverse=ww[0].lane_id>0);keep=[]
  for w in ww:
   if not keep or w.transform.location.distance(keep[-1].transform.location)>1e-3:keep.append(w)
  if len(keep)<2:continue
  w=keep[-1];outs=set()
  for ds in [.1,.5,1.,2.,3.]:
   outs.update(key(n)for n in w.next(ds)if key(n)!=k)
   if outs:break
  adj=[]
  mid=keep[len(keep)//2]
  for n in [mid.get_left_lane(),mid.get_right_lane()]:adj.append(key(n)if n is not None and str(n.lane_type)=='Driving'and n.lane_id*mid.lane_id>0 else None)
  rows.append(dict(key=k,road=w.road_id,section=w.section_id,lane=w.lane_id,block=f'{w.road_id}:{w.section_id}:{1 if w.lane_id>0 else -1}',junction=bool(mid.is_junction),junction_id=int(mid.junction_id),speed_limit_mps=roads.get(w.road_id),path=[[*pose(x.transform).tolist(),float(x.lane_width)]for x in keep],outgoing=sorted(outs),adjacent=adj))
 d=dict(name=cmap.name,opendrive_sha=hashlib.sha256(od.encode()).hexdigest(),coordinate='global x,-y,-yaw; static current map,1m waypoints plus topology endpoints',lanes=rows)
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d));tmp.replace(path);return d
def frame(hero,timestamp,geometry,map_data,route,episode_id,previous=None):
 body=pose(hero.get_transform());R=rot(body[2]);offset=np.asarray(geometry['rear_offset_math']);rear=body[:2]+R@offset
 omega=-math.radians(hero.get_angular_velocity().z);alpha=0.
 if previous is not None:
  dt=timestamp-previous['sim_seconds'];alpha=(omega-previous['omega'])/dt if dt>1e-5 else 0.
 r=R@offset;vel=world_vector(hero.get_velocity())+omega*np.array([-r[1],r[0]]);acc=world_vector(hero.get_acceleration())+alpha*np.array([-r[1],r[0]])-omega*omega*r
 lane_ids={x['key']:str(i+1)for i,x in enumerate(map_data['lanes'])};blocks={k:str(10000000+i)for i,k in enumerate(sorted({x['block']for x in map_data['lanes']}))};lane_by={x['key']:x for x in map_data['lanes']}
 route_ids=[]
 for tr,_ in route:
  w=hero.get_world().get_map().get_waypoint(tr.location);k=key(w)
  if k in lane_by:
   bid=blocks[lane_by[k]['block']]
   if not route_ids or bid!=route_ids[-1]:route_ids.append(bid)
 boxes=[];names=[];vels=[];tokens=[];lights=[]
 for a in hero.get_world().get_actors():
  if a.id==hero.id:continue
  if a.type_id=='traffic.traffic_light':
   red=str(a.get_state())=='Red'
   for wp in a.get_stop_waypoints():
    lk=key(wp)
    if lk in lane_by:
     targets=[lk]if lane_by[lk]['junction']else lane_by[lk]['outgoing']
     for target in targets:
      if target in lane_ids:lights.append((int(lane_ids[target]),red))
   continue
  if not a.type_id.startswith(('vehicle.','walker.pedestrian.','static.prop.'))or not hasattr(a,'bounding_box'):continue
  tf=a.get_transform()
  if np.linalg.norm(pose(tf)[:2]-rear)>100:continue
  box=a.bounding_box;center=tf.transform(box.location);center2=np.array([center.x,-center.y]);xy=R.T@(center2-rear);h=-math.radians(tf.rotation.yaw+box.rotation.yaw)-body[2];h=math.atan2(math.sin(h),math.cos(h))
  if min(box.extent.x,box.extent.y)<=0:continue
  boxes.append([*xy,center.z,box.extent.x*2,box.extent.y*2,box.extent.z*2,h]);vels.append([*(R.T@world_vector(a.get_velocity())),a.get_velocity().z]);tokens.append(f'{episode_id}:actor:{a.id}')
  names.append('vehicle'if a.type_id.startswith('vehicle.')else'pedestrian'if a.type_id.startswith('walker')else'traffic_cone'if'cone'in a.type_id else'barrier'if'barrier'in a.type_id else'generic_object')
 h=float(body[2]);ts=int(round(timestamp*1e6));token=f'{episode_id}:{ts}'
 out=dict(token=token,scene_token=token,log_name=episode_id,map_location=map_data['name'],timestamp=ts,ego2global_translation=[*rear,hero.get_location().z],ego2global_rotation=[math.cos(h/2),0.,0.,math.sin(h/2)],ego_dynamic_state=[*(R.T@vel),*(R.T@acc)],driving_command=[0,0,0,1],roadblock_ids=route_ids,traffic_lights=list(dict(lights).items()),anns=dict(gt_boxes=boxes,gt_names=names,gt_velocity_3d=vels,track_tokens=tokens,instance_tokens=tokens),cams={},lidar_path='',sim_seconds=timestamp,omega=omega,body_pose_math=body.tolist(),rear_geometry=geometry,current_only=True)
 return out
