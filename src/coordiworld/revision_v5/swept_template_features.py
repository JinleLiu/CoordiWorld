"""Pinned array interpolation, avoiding intermediate pose objects; same current paths."""
import numpy as np

def extract_swept_rails(context):
    from .static_indexed_map import indexed_map
    from .route_template_cache import current_manager,sample_template
    from copy import deepcopy
    from .current_rail_response import make_policy
    from navsim.traffic_agents_policies.abstract_traffic_agents_policy import filter_tracked_objects_by_type
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    p=context.map_parameters;mp=get_maps_api(p.map_root,p.map_version,p.map_name);mp=indexed_map(mp,(p.map_root,p.map_version,p.map_name))
    vehicle=filter_tracked_objects_by_type(context.current_tracked_objects,TrackedObjectType.VEHICLE)[0]
    obs=make_policy(p.map_root)._idm_agents_observation
    manager=current_manager(obs,context.ego_state,vehicle,mp)
    origin=context.ego_state.rear_axle;c,s=np.cos(origin.heading),np.sin(origin.heading);rot=np.array([[c,s],[-s,c]])
    tokens=[];paths=[];info=[];initial=[]
    for token,a in manager.agents.items():
        # Preserve every active current actor, including relevant outside-ROI leaders.
        linear,angular,start,end=sample_template(a._path)
        xy=(linear[:,1:3]-np.array([origin.x,origin.y]))@rot.T
        heading=np.unwrap(angular[:,0]-origin.heading)
        paths.append(np.c_[xy,heading]);tokens.append(token)
        box=a._initial_state.box.center;xy0=rot@np.array([box.x-origin.x,box.y-origin.y]);initial.append([xy0[0],xy0[1],box.heading-origin.heading])
        info.append([a.progress-start,end-start,a.velocity,a.length,a.width,a._policy.target_velocity,float(a.velocity<=.5),len(a.end_segment.outgoing_edges)])
    return np.asarray(paths,np.float32).reshape(-1,256,3),np.asarray(info,np.float32).reshape(-1,8),tokens,np.asarray(initial,np.float32).reshape(-1,3)

