"""Current-only map-rail response mechanism, an explicitly same-family baseline.

No MetricCache or teacher trajectory is accepted. Complete current annotations
and the static map extend the compressed 50m/24-token neural input contract.
This is NOT independent causal evidence or a learned-response qualification.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np


def current_context(history, map_root):
    from navsim.common.dataclasses import Scene
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state, annotations_to_detection_tracks
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    if len(history) != 4:
        raise ValueError('Exactly four current/history frames, no future')
    now = history[-1]
    if not np.allclose(np.diff([f['timestamp'] for f in history])/1e6, .5, atol=.08):
        raise ValueError('History is not contiguous')
    ego = ego_status_to_ego_state(Scene._build_ego_status(now), get_pacifica_parameters(), TimePoint(int(now['timestamp'])))
    tracks = annotations_to_detection_tracks(Scene._build_annotations(now), ego)
    name = now['map_location']; name = 'us-nv-las-vegas-strip' if name == 'las_vegas' else name
    return SimpleNamespace(ego_state=ego, current_tracked_objects=[tracks],
        map_parameters=SimpleNamespace(map_root=str(map_root), map_version='nuplan-maps-v1.0', map_name=name),
        future_tracked_objects=[DetectionsTracks(TrackedObjects([])) for _ in range(40)], traffic_light_status=None)


def restrict_context(context, tracks, scene=None):
    """Retain exact tensor actor identities; optionally use tensor order/precision.

Static road graph remains complete in both cases, so this is not a claim that
24 map tokens alone identify the teacher.
"""
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    result = deepcopy(context); selected = set(tracks)
    objects = [o for o in result.current_tracked_objects[0].tracked_objects if o.track_token in selected]
    if scene is not None:
        from nuplan.common.actor_state.agent import Agent
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
        origin = context.ego_state.rear_axle; c,s=np.cos(origin.heading),np.sin(origin.heading)
        rot=np.array([[c,-s],[s,c]]); byid={o.track_token:o for o in objects}; rebuilt=[]
        values=scene.agents[0,-1].numpy()
        for i, token in enumerate(tracks):
            obj=byid.get(token)
            if obj is None: continue
            if obj.tracked_object_type.name != 'VEHICLE':
                rebuilt.append(obj); continue
            v=values[i].astype(np.float64); xy=rot@v[:2]+np.array([origin.x,origin.y]); vel=rot@v[3:5]
            box=OrientedBox(StateSE2(*xy, float(v[2]+origin.heading)),float(v[5]),float(v[6]),obj.box.height)
            rebuilt.append(Agent(obj.tracked_object_type, box, StateVector2D(*vel),obj.metadata))
        objects=rebuilt
    result.current_tracked_objects=[DetectionsTracks(TrackedObjects(objects))]
    return result


def make_policy(map_root):
    from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
    from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    obs=NavsimIDMAgents(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,
        accel_max=1.,decel_max=2.,open_loop_detections_types=[],minimum_path_length=20,
        planned_trajectory_samples=None,planned_trajectory_sample_interval=None,radius=100,
        add_open_loop_parked_vehicles=True,idm_snap_threshold=3.)
    return NavsimIDMTrafficAgents(TrajectorySampling(num_poses=40,interval_length=.1),obs,map_root_override=str(map_root))


def rollout_current(policy, context, full_execution):
    # Only vehicle response is returned; no privileged non-vehicle replay merge.
    if context.traffic_light_status is not None or any(len(o.tracked_objects) for o in context.future_tracked_objects):
        raise ValueError('Future context is forbidden')
    return context.current_tracked_objects + policy.simulate_traffic_agents(full_execution, context)


def compose_reference_preserving(reference_states, delta_states, response_valid, reference):
    """Keep the learned factual reference EXACTLY; unavailable response stays zero.

Unknown response support remains explicitly returned and is not a zero label.
Size is never changed. This algebraic composition needs separate absolute-world,
planning, collision, calibration and latency validation before deployment.
"""
    if np.any(delta_states[reference] != 0): raise ValueError('Reference response must be zero')
    d=np.where(response_valid[...,None],delta_states,0).copy(); d[...,5:]=0
    result=np.broadcast_to(reference_states[None],d.shape).copy()+d
    result[...,2]=np.arctan2(np.sin(result[...,2]),np.cos(result[...,2]))
    result[reference]=reference_states
    assert np.array_equal(result[reference],reference_states)
    return result
