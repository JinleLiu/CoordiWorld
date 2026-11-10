"""Pinned NAVSIM v2 component scorer for original scenes.

MetricCache contains future ground truth and MUST NEVER feed the policy/model.
This wrapper is an evaluation boundary. Full navhard EPDMS additionally needs
the official second-stage scene mapping and two-frame comfort aggregation.
"""
import os
import gc
from pathlib import Path
import numpy as np

NAVSIM_COMMIT='0a380a9063d7162ec93d0f51e9990ebac585f720'
NUPLAN_COMMIT='ce3c323af01c0d7ec5672f7832ef53f9c679aab0'


class OfficialComponentScorer:
    def __init__(self,map_root,cache_root,traffic='non_reactive',map_version='nuplan-maps-v1.0'):
        os.environ['NUPLAN_MAPS_ROOT']=str(Path(map_root).resolve())
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
        self.map_root=str(Path(map_root).resolve());self.map_version=map_version
        self.sampling=TrajectorySampling(num_poses=40,interval_length=.1)
        self.prediction_sampling=TrajectorySampling(num_poses=8,interval_length=.5)
        self.processor=MetricCacheProcessor(str(cache_root),False,self.sampling)
        self.scorer=PDMScorer(self.sampling,PDMScorerConfig(human_penalty_filter=True))
        self.simulator=PDMSimulator(self.sampling)
        if traffic=='non_reactive':self.traffic=LogReplayTrafficAgents(self.sampling)
        elif traffic=='reactive':
            from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
            from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
            observation=NavsimIDMAgents(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=2.,open_loop_detections_types=[],minimum_path_length=20,planned_trajectory_samples=None,planned_trajectory_sample_interval=None,radius=100,add_open_loop_parked_vehicles=True,idm_snap_threshold=3.)
            self.traffic=NavsimIDMTrafficAgents(self.sampling,observation,map_root_override=self.map_root)
        else:raise ValueError(traffic)
        self.traffic_name=traffic

    def build_metric_cache(self,frames,current_index):
        from navsim.common.dataclasses import Scene,SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        history=4;future=8;sequence=frames[current_index-history+1:current_index+future+1]
        if len(sequence)!=history+future:raise ValueError('Not enough frames for official evaluation')
        scene=Scene.from_scene_dict_list(sequence,Path('.'),history,future,SensorConfig.build_no_sensors())
        scenario=NavSimScenario(scene,self.map_root,self.map_version)
        try:return self.processor.compute_metric_cache(scenario)
        finally:gc.enable()

    def score(self,metric_cache,trajectory):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
        table,_=pdm_score(metric_cache,Trajectory(np.asarray(trajectory,dtype=np.float64),self.prediction_sampling),self.sampling,self.simulator,self.scorer,self.traffic)
        row=table.iloc[0]
        weights=np.asarray(row['weighted_metrics_array'],float).copy()
        weights[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT]=0.
        metrics=np.asarray(row['weighted_metrics'],float)
        # Explicit development composite, recomputed after official human filtering.
        composite=float(row['multiplicative_metrics_prod']*np.sum(metrics[weights>0]*weights[weights>0])/weights.sum())
        names=['no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance','ego_progress','time_to_collision_within_bound','lane_keeping','history_comfort','multiplicative_metrics_prod']
        result={name:float(row[name]) for name in names}
        if not all(np.isfinite(v) for v in result.values()) or not np.isfinite(composite):raise ValueError('Non-finite official component')
        result['single_stage_composite_excluding_two_frame_comfort']=composite
        return result
