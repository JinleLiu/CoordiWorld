"""Pinned history-comfort calculation from history and candidate execution only.

No MetricCache, future log frame, traffic teacher, or native outcome is accepted.
This exposes a known ego-only criterion, not a learned risk probability.
"""
from pathlib import Path
import numpy as np

def current_history_comfort(history, full_execution, map_root):
    if len(history)!=4:raise ValueError('Exactly four observed history frames required')
    from navsim.common.dataclasses import Scene,SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import TimePoint,ego_states_to_state_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
    scene=Scene.from_scene_dict_list(history,Path('.'),4,0,SensorConfig.build_no_sensors())
    scenario=NavSimScenario(scene,str(map_root),'nuplan-maps-v1.0')
    past=InterpolatedTrajectory(list(scenario.get_ego_past_trajectory(0,1.5)))
    start,end=past.start_time.time_us,past.end_time.time_us
    times=np.clip(np.arange(start,end,100000,dtype=np.int64),start,end)
    states=ego_states_to_state_array(past.get_state_at_times([TimePoint(int(t)) for t in times[:-1]]))
    ids=sorted(full_execution);execution=np.stack([full_execution[j] for j in ids]);padded=np.concatenate((np.broadcast_to(states,(len(ids),*states.shape)),execution),axis=1)
    flags=ego_is_comfortable(padded,np.arange(padded.shape[1])*.1)
    score=np.ones(12,np.float32);score[ids]=flags.all(-1)
    return score,dict(valid_ids=ids,component_pass=flags.tolist(),past_sample_count=len(states),input_latest_timestamp=int(history[-1]['timestamp']),future_log_frames=0)
