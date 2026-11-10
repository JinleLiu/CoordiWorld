"""Current-frame nuPlan map adapter. No future traffic-light states are read.

The hard labels here are transparent development proxies, not NAVSIM metrics.
Route containment is a route-corridor proxy; individual lane tokens provide
context without incorrectly requiring a vehicle to be inside every lane.
"""
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Point,Polygon
from shapely.ops import unary_union
from shapely import make_valid
from .openscene import ego_pose,transform_pose,PACIFICA_LENGTH,PACIFICA_WIDTH,PACIFICA_REAR_AXLE_TO_CENTER
from ..geometry import box_polygon


def polygonal_geometry(geometry):
    geometry=geometry if geometry.is_valid else make_valid(geometry)
    if geometry.geom_type in ('Polygon','MultiPolygon'):return geometry
    pieces=[]
    for part in getattr(geometry,'geoms',[]):
        polygon=polygonal_geometry(part)
        if not polygon.is_empty:pieces.append(polygon)
    return unary_union(pieces) if pieces else Polygon()


def local_geometry(geometry,origin):
    c,s=np.cos(origin[2]),np.sin(origin[2]);x,y=origin[:2]
    transformed=affine_transform(geometry,[c,s,-s,c,-c*x-s*y,s*x-c*y])
    # Official polygons can contain self-intersections, and rigid transforms can
    # expose nearly coincident boundary topology. Preserve all valid components.
    return polygonal_geometry(transformed)


def geometry_features(geometry,heading=0.,signal=0.,signal_valid=False):
    if geometry.is_empty:return [0.]*8
    center=geometry.centroid;left,bottom,right,top=geometry.bounds
    return [center.x,center.y,float(heading),right-left,top-bottom,float(signal),float(signal_valid),0.]


class CurrentMapProvider:
    def __init__(self,map_root,map_version='nuplan-maps-v1.0',radius=100.):
        self.map_root=str(map_root);self.map_version=map_version;self.radius=radius

    def __call__(self,history,max_tokens=24):
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer,StopLineType
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        frame=history[-1];origin=ego_pose(frame)
        api=get_maps_api(self.map_root,self.map_version,frame['map_location'])
        layers=[Layer.ROADBLOCK,Layer.ROADBLOCK_CONNECTOR,Layer.INTERSECTION,Layer.CARPARK_AREA,Layer.STOP_LINE]
        objects=api.get_proximal_map_objects(Point2D(*origin[:2]),self.radius,layers)
        roadblocks=objects[Layer.ROADBLOCK]+objects[Layer.ROADBLOCK_CONNECTOR]
        edges={str(edge.id):edge for block in roadblocks for edge in block.interior_edges}
        drivable=[o.polygon for o in objects[Layer.ROADBLOCK]+objects[Layer.INTERSECTION]+objects[Layer.CARPARK_AREA]]
        drivable.extend(edge.polygon for edge in edges.values())
        route_ids=set(map(str,frame['roadblock_ids']))
        route=[edge.polygon for block in roadblocks if str(block.id) in route_ids for edge in block.interior_edges]
        tokens=[]
        for kind,polygons in [('drivable',drivable),('lane',route)]:
            if polygons:
                polygons=[polygonal_geometry(poly) for poly in polygons]
                geometry=local_geometry(unary_union(polygons),origin)
                tokens.append({'kind':kind,'geometry':geometry,'features':geometry_features(geometry),'id':kind+'_union','semantic_scope':'drivable_polygon_union' if kind=='drivable' else 'route_corridor_containment_proxy'})
        controls=[]
        for connector,is_red in frame['traffic_lights']:
            edge=edges.get(str(connector))
            if edge is None:continue
            geometry=local_geometry(edge.polygon,origin)
            controls.append({'kind':'signal','geometry':geometry,'features':geometry_features(geometry,signal=float(is_red),signal_valid=True),'id':'signal_'+str(connector),'signal_valid':True,'prohibited':bool(is_red),'semantic_scope':'current_red_connector_overlap'})
        for stop in objects[Layer.STOP_LINE]:
            # Traffic-light stop lines are handled through their current signal;
            # yield/crosswalk/unknown polygons do not imply a mandatory stop.
            if int(stop.stop_line_type)!=int(StopLineType.STOP_SIGN):continue
            geometry=local_geometry(stop.polygon,origin)
            satisfied=False
            for past in history:
                pose=transform_pose(ego_pose(past),origin)
                footprint=box_polygon(pose,PACIFICA_LENGTH,PACIFICA_WIDTH,PACIFICA_REAR_AXLE_TO_CENTER)
                speed=np.linalg.norm(past['ego_dynamic_state'][:2])
                satisfied |= bool(speed<.3 and footprint.distance(geometry)<2.)
            controls.append({'kind':'stop','geometry':geometry,'stop_line':geometry,'features':geometry_features(geometry),'id':'stop_'+str(stop.id),'prohibited':True,'initially_stopped':satisfied,'dt':.5,'semantic_scope':'observed_stop_before_stop_polygon_crossing'})
        controls.sort(key=lambda t:(t['geometry'].distance(Point(0,0)),t['id']))
        tokens.extend(controls)
        contexts=[]
        for identifier,edge in edges.items():
            geometry=local_geometry(edge.polygon,origin)
            path=edge.baseline_path.discrete_path
            heading=float(np.arctan2(np.sin(path[0].heading-origin[2]),np.cos(path[0].heading-origin[2]))) if path else 0.
            contexts.append({'kind':'lane','geometry':geometry,'features':geometry_features(geometry,heading),'id':'lane_context_'+identifier,'rule_enabled':False,'semantic_scope':'context_only'})
        contexts.sort(key=lambda t:(t['geometry'].distance(Point(0,0)),t['id']))
        tokens.extend(contexts)
        selected=tokens[:max_tokens]
        for token in selected:token['available']=not token['geometry'].is_empty
        return selected
