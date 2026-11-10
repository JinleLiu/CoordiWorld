"""Explicit rear-axle correction for a versioned IDM teacher domain.

The pinned IDM builder interprets array X/Y as body center, while PDMSimulator
emits rear-axle X/Y. This wrapper changes only that conversion boundary. It
does not alter controller arrays, candidates, timing or pinned source files.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


DOMAIN = "privileged_NAVSIM_IDM_rear_axle_corrected_endpoint_compat_v2"


def rear_axle_array_for_center_builder(state, offset):
    state = np.asarray(state)
    if state.shape != (11,) or not np.isfinite(state).all() or not math.isfinite(offset) or offset <= 0:
        raise ValueError("Expected finite pinned 11-channel rear-axle state and positive body offset")
    converted = state.astype(np.float64, copy=True)
    converted[:2] += float(offset) * np.array([math.cos(float(state[2])), math.sin(float(state[2]))])
    return converted


def wrap_center_builder(original, stats=None):
    stats = stats if stats is not None else {}
    stats.update(calls=0, max_rear_axle_error_m=0., max_heading_error_rad=0.)
    def corrected(ego_state_arr, initial_ego_state, time_point):
        source = np.asarray(ego_state_arr)
        offset = float(initial_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center)
        converted = rear_axle_array_for_center_builder(source, offset)
        result = original(converted, initial_ego_state, time_point)
        actual = result.rear_axle
        error = float(np.linalg.norm(np.array([actual.x, actual.y]) - source[:2]))
        yaw_error = abs(math.atan2(math.sin(actual.heading - source[2]), math.cos(actual.heading - source[2])))
        if error > 1e-7 or yaw_error > 1e-10:
            raise ValueError("Corrected IDM builder does not preserve the supplied rear-axle pose")
        stats["calls"] += 1
        stats["max_rear_axle_error_m"] = max(stats["max_rear_axle_error_m"], error)
        stats["max_heading_error_rad"] = max(stats["max_heading_error_rad"], yaw_error)
        return result
    corrected._coordiworld_teacher_domain = DOMAIN
    return corrected


def audit_native_builder():
    """Run real pinned CarFootprint/IDM construction, without simulation/data."""
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.actor_state.state_representation import TimePoint
    from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import build_ego_state_from_simulated_ego_states
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    if (StateIndex.X, StateIndex.Y, StateIndex.HEADING) != (0, 1, 2):
        raise ValueError("Pinned state index contract changed")
    original = build_ego_state_from_simulated_ego_states
    if hasattr(original, "_coordiworld_teacher_domain"):
        raise ValueError("Geometry audit must precede installation")
    params = get_pacifica_parameters()
    initial = SimpleNamespace(car_footprint=SimpleNamespace(vehicle_parameters=params), is_in_auto_mode=True)
    corrected = wrap_center_builder(original)
    rows = []
    for heading in (0., math.pi / 2, -2.1):
        state = np.zeros(11, dtype=np.float64); state[:3] = [100., 200., heading]
        old = original(state, initial, TimePoint(100000))
        new = corrected(state, initial, TimePoint(100000))
        def xy(value): return [float(value.x), float(value.y)]
        old_error = float(np.linalg.norm(np.asarray(xy(old.rear_axle)) - state[:2]))
        new_error = float(np.linalg.norm(np.asarray(xy(new.rear_axle)) - state[:2]))
        if abs(old_error - params.rear_axle_to_center) > 1e-7 or new_error > 1e-7:
            raise ValueError("Observed native builder does not match the audited mismatch/correction")
        rows.append({"heading": heading, "supplied_rear_axle": state[:2].tolist(),
            "original_center": xy(old.center), "original_rear_axle": xy(old.rear_axle),
            "corrected_center": xy(new.center), "corrected_rear_axle": xy(new.rear_axle),
            "original_rear_axle_error_m": old_error, "corrected_rear_axle_error_m": new_error})
    return {"teacher_domain": DOMAIN, "native_runtime_audited": True,
        "rear_axle_to_center_m": float(params.rear_axle_to_center), "rows": rows,
        "controller_candidates_and_arrays_changed": False, "timing_changed": False,
        "pinned_source_files_modified": False}


def install_corrected_builder():
    import navsim.traffic_agents_policies.navsim_IDM_traffic_agents as module
    original = module.build_ego_state_from_simulated_ego_states
    if hasattr(original, "_coordiworld_teacher_domain"):
        raise ValueError("Refuse duplicate geometry patch")
    audit = audit_native_builder(); stats = {}
    module.build_ego_state_from_simulated_ego_states = wrap_center_builder(original, stats)
    return audit, stats
