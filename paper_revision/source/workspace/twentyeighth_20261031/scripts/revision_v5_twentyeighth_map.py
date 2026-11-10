"""Small nuPlan map interface backed by CARLA's actual static lane topology."""
import numpy as np
from types import SimpleNamespace
from shapely.geometry import Point,LineString,Polygon
from shapely.ops import unary_union
from shapely import make_valid
from shapely.strtree import STRtree
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
class Baseline:
 def __init__(self,path):
  self.a=np.asarray(path)[:,:3];self.discrete_path=[StateSE2(*r)for r in self.a];self.linestring=LineString(self.a[:,:2]);self.length=self.linestring.length
 def get_nearest_arc_length_from_position(self,p):return self.linestring.project(Point(p.x,p.y))
 def get_nearest_pose_from_position(self,p):
  s=self.get_nearest_arc_length_from_position(p);q=self.linestring.interpolate(s);arc=np.r_[0,np.linalg.norm(np.diff(self.a[:,:2],axis=0),axis=1).cumsum()];h=np.interp(s,arc,np.unwrap(self.a[:,2]));return StateSE2(q.x,q.y,float(h))
 def get_curvature_at_arc_length(self,s):
  arc=np.r_[0,np.linalg.norm(np.diff(self.a[:,:2],axis=0),axis=1).cumsum()];return float(np.interp(s,arc,np.gradient(np.unwrap(self.a[:,2]),arc)))
class Lane:
 def __init__(self,r,i):
  self.id=str(i);self.row=r;self.baseline_path=Baseline(r['path']);a=np.asarray(r['path']);n=np.c_[-np.sin(a[:,2]),np.cos(a[:,2])]*a[:,3:4]/2;self.polygon=make_valid(Polygon(np.r_[a[:,:2]+n,(a[:,:2]-n)[::-1]]));self.speed_limit_mps=r['speed_limit_mps'];self.outgoing_edges=[];self.incoming_edges=[];self.stop_lines=[];self.index=1
 def get_roadblock_id(self):return self.parent.id
 def get_roadblock(self):return self.parent
 def contains_point(self,p):return self.polygon.covers(Point(p.x,p.y))
 def get_distance_to_nearest_map_object(self,p):return self.polygon.distance(Point(p.x,p.y))
 def has_traffic_lights(self):return bool(self.stop_lines)
 @property
 def is_left_of(self):return False
class Block:
 def __init__(self,i,lanes):
  self.id=str(i);self.interior_edges=lanes;self.polygon=unary_union([l.polygon for l in lanes]);self.outgoing_edges=[];self.incoming_edges=[];self.stop_lines=[];self.parallel_edges=[]
  for j,l in enumerate(lanes):l.parent=self;l.index=j+1
 def contains_point(self,p):return self.polygon.covers(Point(p.x,p.y))
class CarlaMap:
 _LANE_CONNECTOR_POLYGON_LAYER='lane_connector_polygons'
 def __init__(self,data):
  self.map_name=data['name'];self.map_version='CARLA0.9.15';self.lanes={r['key']:Lane(r,i+1)for i,r in enumerate(data['lanes'])};groups={}
  for x in self.lanes.values():groups.setdefault(x.row['block'],[]).append(x)
  self.blocks={k:Block(10000000+i,groups[k])for i,k in enumerate(sorted(groups))}
  for x in self.lanes.values():
   x.outgoing_edges=[self.lanes[k]for k in x.row['outgoing']if k in self.lanes];x.adjacent_edges=tuple(self.lanes.get(k)for k in x.row['adjacent'])
   for y in x.outgoing_edges:y.incoming_edges.append(x)
  for x in self.blocks.values():
   x.outgoing_edges=list({l.parent.id:l.parent for a in x.interior_edges for l in a.outgoing_edges if l.parent is not x}.values());x.incoming_edges=list({l.parent.id:l.parent for a in x.interior_edges for l in a.incoming_edges if l.parent is not x}.values())
  self.layers={l:[]for l in L};self.layers[L.LANE]=[x for x in self.lanes.values()if not x.row['junction']];self.layers[L.LANE_CONNECTOR]=[x for x in self.lanes.values()if x.row['junction']];self.layers[L.ROADBLOCK]=[x for x in self.blocks.values()if not x.interior_edges[0].row['junction']];self.layers[L.ROADBLOCK_CONNECTOR]=[x for x in self.blocks.values()if x.interior_edges[0].row['junction']]
  js={}
  for x in self.lanes.values():
   if x.row['junction']:js.setdefault(x.row['junction_id'],[]).append(x.polygon)
  self.layers[L.INTERSECTION]=[SimpleNamespace(id=str(20000000+k),polygon=unary_union(v))for k,v in js.items()];self.lookup={l:{x.id:x for x in v}for l,v in self.layers.items()};self.frames={};self.trees={l:STRtree([x.polygon for x in v])for l,v in self.layers.items()if v}
 def get_map_object(self,i,layer):return self.lookup.get(layer,{}).get(str(i))
 def _get_roadblock(self,i):return self.get_map_object(i,L.ROADBLOCK)
 def _get_roadblock_connector(self,i):return self.get_map_object(i,L.ROADBLOCK_CONNECTOR)
 def get_proximal_map_objects(self,point,radius,layers):
  p=Point(point.x,point.y);out={}
  for l in layers:
   v=self.layers.get(l,[]);ids=sorted(self.trees[l].query(p,predicate='dwithin',distance=radius))if l in self.trees else[];out[l]=[v[i]for i in ids if v[i].polygon.distance(p)<=radius]
  return out
 def get_all_map_objects(self,point,layer):
  p=Point(point.x,point.y);v=self.layers.get(layer,[]);ids=sorted(self.trees[layer].query(p,predicate='intersects'))if layer in self.trees else[];return [v[i]for i in ids if v[i].polygon.covers(p)]
 def is_in_layer(self,point,layer):return bool(self.get_all_map_objects(point,layer))
 def get_distance_to_nearest_map_object(self,point,layer):
  p=Point(point.x,point.y);v=self.layers.get(layer,[])
  if not v:return None,float('inf')
  tree=self.trees[layer];j=int(tree.nearest(p));d=v[j].polygon.distance(p);ids=sorted(tree.query(p,predicate='dwithin',distance=d+1e-7));i=min(ids,key=lambda i:v[i].polygon.distance(p));return v[i].id,v[i].polygon.distance(p)
 def _get_vector_map_layer(self,layer):
  if layer not in self.frames:
   import geopandas as gpd
   v=self.layers.get(layer,[]);self.frames[layer]=gpd.GeoDataFrame({'fid':[x.id for x in v],'lane_connector_fid':[x.id for x in v]},geometry=[x.polygon for x in v])
  return self.frames[layer]
 def _load_vector_map_layer(self,name):return self._get_vector_map_layer(L.LANE_CONNECTOR if name==self._LANE_CONNECTOR_POLYGON_LAYER else name)
 def get_available_map_objects(self):return list(self.layers)
def install(data):
 import sys
 import nuplan.common.maps.nuplan_map.map_factory as f
 import navsim.common.dataclasses as d
 import navsim.planning.scenario_builder.navsim_scenario as n
 mp=CarlaMap(data);old=f.get_maps_api
 def getter(root,version,name):
  if name==mp.map_name:return mp
  return old(root,version,name)
 for module in list(sys.modules.values()):
  if module is not None and getattr(module,'get_maps_api',None)is old:setattr(module,'get_maps_api',getter)
 d.Scene._build_map_api=classmethod(lambda cls,name:mp if name==mp.map_name else getter('', '',name))
 n.NavSimScenario.map_api=property(lambda self:mp if self._map_name==mp.map_name else getter(self._map_root,self._map_version,self._map_name))
 return mp
