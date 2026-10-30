"""Static polygon broad phase with native strict contains and original row order.

No actor state, route selection, controller execution or prediction is cached.
This wrapper is for the pinned current-rail builder, not arbitrary map mutation.
"""
from shapely.geometry import Point
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


class StaticIndexedMap:
    def __init__(self,native):
        self.native=native;self.layers={}

    def __getattr__(self,name):
        return getattr(self.native,name)

    def _query(self,point,layer,connector_polygons=False):
        key=(layer,connector_polygons)
        if key not in self.layers:
            frame=(self.native._load_vector_map_layer(self.native._LANE_CONNECTOR_POLYGON_LAYER)
                   if connector_polygons else self.native._get_vector_map_layer(layer))
            geometries=frame.geometry.to_numpy()
            column='lane_connector_fid' if connector_polygons else 'fid'
            ids=frame[column].tolist() if column in frame.columns else [None]*len(frame)
            self.layers[key]=(frame.sindex,geometries,ids)
        index,geometries,ids=self.layers[key];pt=Point(point.x,point.y)
        positions=sorted(set(map(int,index.query(pt))))
        return [ids[j] for j in positions if geometries[j].contains(pt)]

    def is_in_layer(self,point,layer):
        if layer in [SemanticMapLayer.LANE,SemanticMapLayer.INTERSECTION]:
            return bool(self._query(point,layer))
        return self.native.is_in_layer(point,layer)

    def get_all_map_objects(self,point,layer):
        if layer==SemanticMapLayer.LANE:
            return [self.native.get_map_object(i,layer) for i in self._query(point,layer)]
        if layer==SemanticMapLayer.LANE_CONNECTOR:
            return [self.native.get_map_object(str(i),layer) for i in self._query(point,layer,True)]
        return self.native.get_all_map_objects(point,layer)


_STATIC_MAPS={}


def indexed_map(native,key):
    # Include object identity as well as map root/version/name to prevent stale
    # index reuse if a caller intentionally opens another map object.
    cache_key=(tuple(key),id(native))
    if cache_key not in _STATIC_MAPS:_STATIC_MAPS[cache_key]=StaticIndexedMap(native)
    return _STATIC_MAPS[cache_key]
