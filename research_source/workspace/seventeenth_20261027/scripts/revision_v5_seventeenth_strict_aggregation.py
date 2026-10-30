"""Complete pinned NAVSIM aggregation; any incomplete/invalid row aborts output."""
import ast
import hashlib
from pathlib import Path
from typing import Dict,List,Tuple
import numpy as np
import pandas as pd
import navsim
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex, WeightedMetricIndex

# The runner imports an unused training agent/framework. Execute only its four
# unchanged pure aggregation functions, bound to the exact official source SHA.
# No aggregation body, metric definition or numerical constant is rewritten.
_path=Path(navsim.__file__).parent/'planning/script/run_pdm_score.py'
_source=_path.read_bytes()
if hashlib.sha256(_source).hexdigest()!='c0a3a0648b72178203a8d6559a41e8993e81860d9e0367eac91dd94c26a06ecf':
    raise ValueError('official runner source mismatch')
_names={'create_scene_aggregators','compute_final_scores','calculate_individual_mapping_scores','calculate_weighted_average_score'}
_nodes=[n for n in ast.parse(_source).body if isinstance(n,ast.FunctionDef) and n.name in _names]
if {n.name for n in _nodes}!=_names:raise ValueError('missing official pure function')
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(_path),'exec'),globals())

METRICS=['no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance',
         'traffic_light_compliance','ego_progress','time_to_collision_within_bound',
         'lane_keeping','history_comfort','two_frame_extended_comfort','score']

def aggregate_complete(rows,mappings,sampling):
    if not mappings or any(not pairs for pairs in mappings.values()):
        raise ValueError('complete nonempty two-stage mappings required')
    pairs=[pair for key,second in mappings.items() for pair in [key,*second]]
    if any(len(p)!=2 or p[0]==p[1] for p in pairs):
        raise ValueError('invalid frame pairing')
    expected={t for p in pairs for t in p}
    if rows['token'].duplicated().any() or set(rows['token'])!=expected:
        raise ValueError('missing, extra or duplicate scene row')
    if not rows['valid'].eq(True).all():
        raise ValueError('failed scene prevents complete aggregation')
    scalars=METRICS[:8]+['multiplicative_metrics_prod','start_time','endpoint_x','endpoint_y','start_point_x','start_point_y']
    if not np.isfinite(rows[scalars].to_numpy(float)).all():
        raise ValueError('nonfinite component or geometry')
    index=rows.set_index('token')
    for _,r in rows.iterrows():
        states=np.asarray(r['ego_simulated_states'],float)
        if states.shape!=(sampling.num_poses+1,StateIndex.size()) or not np.isfinite(states).all():
            raise ValueError('invalid executed-state sequence')
        w=np.asarray(r['weighted_metrics_array'],float);v=np.asarray(r['weighted_metrics'],float)
        k=int(WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT)
        if w.shape!=(len(WeightedMetricIndex),) or v.shape!=w.shape:
            raise ValueError('invalid weighted-metric shape')
        if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0 or not np.isfinite(np.delete(v,k)).all():
            raise ValueError('invalid weighted metric')
    for now,prev in pairs:
        delta=float(index.loc[now,'start_time']-index.loc[prev,'start_time'])
        offset=round(delta/sampling.interval_length)
        if not 0<delta<.55 or not 0<offset<sampling.num_poses+1:
            raise ValueError('invalid frame time overlap')
    scored=compute_final_scores(create_scene_aggregators(mappings,rows.copy(deep=True),sampling))
    if not np.isfinite(scored[METRICS+['weight']].to_numpy(float)).all():
        raise ValueError('nonfinite complete aggregate')
    combined,first,second=calculate_individual_mapping_scores(scored[METRICS+['token','weight']],mappings)
    if not all(np.isfinite(v.to_numpy(float)).all() for v in [combined,first,second]):
        raise ValueError('nonfinite stage summary')
    return scored,dict(combined=combined.to_dict(),stage_one=first.to_dict(),stage_two=second.to_dict())
