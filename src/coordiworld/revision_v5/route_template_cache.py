"""Bounded reuse of immutable map paths, never actor state or predictions.

The pinned builder's bytecode is reused with only its IDMAgent constructor
dependency replaced. Its original module and other callers remain untouched.
"""
from collections import OrderedDict
from types import FunctionType
import numpy as np
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent

_PATHS=OrderedDict()
_SAMPLES=OrderedDict()
_BUILDERS={}
_LIMIT=4096
_STATS={'path_hits':0,'path_misses':0,'sample_hits':0,'sample_misses':0}


class TemplateIDMAgent(IDMAgent):
    def _convert_route_to_path(self):
        import nuplan.planning.simulation.observation.idm.idm_agent as native
        route=tuple(self._route)
        key=(id(native.create_path_from_se2),tuple((type(e).__name__,str(e.id),id(e)) for e in route))
        if key in _PATHS:
            _STATS['path_hits']+=1;_PATHS.move_to_end(key)
            return _PATHS[key][1]
        _STATS['path_misses']+=1;path=super()._convert_route_to_path()
        _PATHS[key]=(route,path)  # Strong references prevent identity reuse.
        if len(_PATHS)>_LIMIT:_PATHS.popitem(last=False)
        return path


def current_manager(observation,ego,vehicles,map_api):
    from navsim.planning.simulation.observation.navsim_idm.navsim_idm_agents_builder import build_idm_agents_on_map_rails as original
    from navsim.planning.simulation.observation.navsim_idm.navsim_idm_agent_manager import NavsimIDMAgentManager
    key=id(original)
    if key not in _BUILDERS:
        namespace=dict(original.__globals__);namespace['IDMAgent']=TemplateIDMAgent
        _BUILDERS[key]=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    agents,occupancy=_BUILDERS[key](observation._target_velocity,observation._min_gap_to_lead_agent,
        observation._headway_time,observation._accel_max,observation._decel_max,
        observation._minimum_path_length,observation._idm_snap_threshold,
        observation._open_loop_detections_types,ego,vehicles,map_api)
    return NavsimIDMAgentManager(agents,occupancy,map_api)


def sample_template(path):
    key=id(path)
    if key in _SAMPLES:
        _STATS['sample_hits']+=1;_SAMPLES.move_to_end(key)
        return _SAMPLES[key][1:]
    _STATS['sample_misses']+=1;start,end=path.get_start_progress(),path.get_end_progress()
    if not end>start:raise ValueError('Degenerate rail')
    progress=np.linspace(start,end,256);path._assert_progress(float(progress[0]));path._assert_progress(float(progress[-1]))
    linear=path._function_interp_linear(progress);angular=path._angular_interpolator.interpolate(progress)
    linear.flags.writeable=False;angular.flags.writeable=False
    _SAMPLES[key]=(path,linear,angular,start,end)
    if len(_SAMPLES)>_LIMIT:_SAMPLES.popitem(last=False)
    return linear,angular,start,end


def cache_statistics():
    return {**_STATS,'path_entries':len(_PATHS),'sample_entries':len(_SAMPLES),'capacity_each':_LIMIT,
            'cached_actor_states':0,'cached_predictions':0}
