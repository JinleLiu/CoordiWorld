"""Full current-map bundle, without MetricCache, future frames, or planning.

The native query is bounded to 100 m (the frozen proposer radius). It is complete
for that query, not a claim of citywide coverage. Outside-envelope or unresolved
routes remain explicitly unsupported. All polygons retain their original WKB.
"""
import hashlib
import json
import numpy as np
from static_components import CurrentMapProvenance, evaluate_static


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def export_current_map(current, map_root, executed_local):
    # The caller must pass a whitelist record, never a frame/MetricCache object.
    allowed = {'scene_id', 'timestamp_us', 'latest_source_timestamp_us', 'map_name',
               'rear_axle_pose', 'velocity', 'acceleration', 'route_roadblock_ids',
               'traffic_lights', 'source_container_sha256'}
    if set(current) != allowed or current['latest_source_timestamp_us'] > current['timestamp_us']:
        raise ValueError('Current-only whitelist or timestamp boundary violated')
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

    class GeometryOnly(AbstractPDMPlanner):
        def initialize(self, initialization):
            raise RuntimeError('Use explicit current geometry initialization')
        def compute_planner_trajectory(self, current_input):
            raise RuntimeError('Planning is forbidden in the geometry exporter')
        def name(self):
            return 'GeometryOnly'
        def observation_type(self):
            raise RuntimeError('Traffic observations are not accepted')

    vehicle = get_pacifica_parameters()
    origin = np.asarray(current['rear_axle_pose'], dtype=np.float64)
    ego = EgoState.build_from_rear_axle(StateSE2(*origin), StateVector2D(*current['velocity']),
        StateVector2D(*current['acceleration']), 0., TimePoint(current['timestamp_us']), vehicle)
    map_name = 'us-nv-las-vegas-strip' if current['map_name'] == 'las_vegas' else current['map_name']
    planner = GeometryOnly(100.)
    planner._map_api = get_maps_api(str(map_root), 'nuplan-maps-v1.0', map_name)
    planner._load_route_dicts(current['route_roadblock_ids'])
    planner._route_roadblock_correction(ego)
    native_map = PDMDrivableMap.from_simulation(planner._map_api, ego, 100.)
    planner._drivable_area_map = native_map
    lane = planner._get_starting_lane(ego)
    if lane is None:
        raise ValueError('No current route lane: unsupported, no inferred safety')
    roadblocks = list(planner._route_roadblock_dict.values())
    rbids = list(planner._route_roadblock_dict)
    if lane.get_roadblock_id() not in rbids:
        raise ValueError('Starting lane is outside current route')
    window = roadblocks[rbids.index(lane.get_roadblock_id()):][:30]
    route_plan, found = Dijkstra(lane, list(planner._route_lane_dict)).search(window[-1])
    path = PDMPath([p for edge in route_plan for p in edge.baseline_path.discrete_path])
    geometry = [dict(token=str(t), layer=k.name, wkb_hex=g.wkb_hex)
                for t, k, g in zip(native_map._tokens, native_map._map_types, native_map._geometries)]
    route = dict(original_roadblock_ids=list(current['route_roadblock_ids']),
                 corrected_roadblock_ids=rbids, lane_ids=list(planner._route_lane_dict),
                 starting_lane_id=lane.id, searched_roadblocks=[x.id for x in window],
                 dijkstra_path_found=bool(found), chosen_lane_ids=[x.id for x in route_plan],
                 centerline_xy=np.asarray(path.linestring.coords).tolist(),
                 lanes=[dict(token=str(k), wkb_hex=x.polygon.wkb_hex,
                             baseline_xy=[[p.x,p.y] for p in x.baseline_path.discrete_path],
                             speed_limit_mps=x.speed_limit_mps,
                             outgoing_ids=[z.id for z in x.outgoing_edges])
                        for k,x in planner._route_lane_dict.items()])
    poses = np.asarray(executed_local, dtype=np.float64).copy()
    if poses.ndim != 3 or poses.shape[1:] != (41,3) or not np.isfinite(poses).all():
        raise ValueError('Expected saved causal 41-step executed rear-axle poses')
    co,si = np.cos(origin[2]),np.sin(origin[2])
    poses[...,:2] = poses[...,:2] @ np.array([[co,si],[-si,co]]) + origin[:2]
    poses[...,2] += origin[2]
    coords = footprint(poses, vehicle.length, vehicle.width, vehicle.rear_axle_to_center)
    query_center = np.array([ego.center.x,ego.center.y])
    radius = float(np.linalg.norm(coords-query_center,axis=-1).max())
    provenance = CurrentMapProvenance(current['scene_id'],current['timestamp_us'],
        current['latest_source_timestamp_us'],canonical_hash(geometry),canonical_hash(route),
        True,bool(found),radius <= 100.,False)
    bundle = dict(schema='current_native_map_v1', current=current, polygons=geometry, route=route,
        provenance=vars(provenance), query=dict(radius_m=100.,center_xy=query_center.tolist(),
        maximum_executed_corner_distance_m=radius, all_query_polygons_retained=True),
        coordinate_frame='native global map; current rear axle stored explicitly',
        vehicle=dict(length=vehicle.length,width=vehicle.width,rear_axle_to_center=vehicle.rear_axle_to_center),
        semantics='Unfiltered static intermediates. No future lights, human filter, traffic response or normalized official progress.')
    return bundle, coords, (native_map,path,list(planner._route_lane_dict),vehicle,poses)


def footprint(poses, length, width, offset):
    poses=np.asarray(poses,dtype=np.float64)
    forward=np.stack([np.cos(poses[...,2]),np.sin(poses[...,2])],-1)
    left=np.stack([-np.sin(poses[...,2]),np.cos(poses[...,2])],-1)
    center=poses[...,:2]+offset*forward
    return np.stack([center+sx*length/2*forward+sy*width/2*left
                     for sx,sy in [(1,1),(-1,1),(-1,-1),(1,-1)]]+[center],-2)


def from_serialized(bundle, coords):
    import shapely
    if canonical_hash(bundle['polygons']) != bundle['provenance']['geometry_sha256'] or canonical_hash(bundle['route']) != bundle['provenance']['route_sha256']:
        raise ValueError('Serialized geometry/route hash mismatch')
    p=CurrentMapProvenance(**bundle['provenance']);p.validate()
    corners=np.zeros(coords.shape[:2]+(4,),bool);onroute=np.zeros(coords.shape[:2],bool);intersection=np.zeros_like(onroute)
    routeids=set(bundle['route']['lane_ids'])
    for record in bundle['polygons']:
        poly=shapely.from_wkb(bytes.fromhex(record['wkb_hex']))
        # Native corners use vectorized.contains; native intersection uses
        # point-within-polygon. Both exclude the boundary; covers is incorrect.
        inside=shapely.contains_xy(poly,coords[...,0],coords[...,1])
        if record['layer'] in {'ROADBLOCK','INTERSECTION','DRIVABLE_AREA','CARPARK_AREA'}:
            corners |= inside[...,:4]
        if record['layer'] in {'LANE','LANE_CONNECTOR'} and record['token'] in routeids:
            onroute |= inside[...,4]
        if record['layer']=='INTERSECTION':
            intersection |= inside[...,4]
    result=evaluate_static(coords[...,4,:],corners,onroute,intersection,bundle['route']['centerline_xy'],p)
    return result,dict(corners=corners,onroute=onroute,intersection=intersection)
