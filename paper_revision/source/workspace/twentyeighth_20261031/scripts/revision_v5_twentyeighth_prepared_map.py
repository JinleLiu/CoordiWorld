"""Exact map predicates: prepare immutable polygons, query only the requested layer.

within(point, polygon) == contains(polygon, point), including boundary exclusion.
No buffering, coordinate rounding, topology change or metric-weight change.
"""
import numpy as np
import shapely
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
ORIGINAL_INIT=PDMDrivableMap.__init__;ORIGINAL_QUERY=PDMDrivableMap.is_in_layer
STATS=dict(calls=0,verified=0,prepared_maps=0);VERIFY=False
def init(self,*args,**kwargs):
 ORIGINAL_INIT(self,*args,**kwargs);shapely.prepare(self._geometries);self._r18_layers={};STATS['prepared_maps']+=1
def query(self,point,layer):
 if layer not in self._r18_layers:self._r18_layers[layer]=np.asarray([g for g,k in zip(self._geometries,self._map_types)if k==layer],dtype=object)
 g=self._r18_layers[layer];value=bool(np.any(shapely.contains_xy(g,point.x,point.y)));STATS['calls']+=1
 if VERIFY:
  assert value==ORIGINAL_QUERY(self,point,layer),(point,layer);STATS['verified']+=1
 return value
def install(verify=False):
 global VERIFY
 VERIFY=verify;PDMDrivableMap.__init__=init;PDMDrivableMap.is_in_layer=query
def restore():PDMDrivableMap.__init__=ORIGINAL_INIT;PDMDrivableMap.is_in_layer=ORIGINAL_QUERY
