"""Float64 compiled execution of the pinned structured LQR/bicycle loop.

Native interpolation and velocity/curvature fitting remain unchanged. New
candidate states are read every call. No global monkeypatch or result cache.
"""
import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from .structured_lqr import StructuredPDMSimulator

_FUNCTION = None


def kernel():
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    source = Path(__file__).with_name('compiled_controller.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = Path(tempfile.gettempdir()) / 'coordiworld_v5_controller'
    directory.mkdir(exist_ok=True)
    library = directory / ('controller_' + digest[:16] + '.so')
    if not library.exists():
        command = ['g++', '-O3', '-std=c++17', '-fPIC', '-shared', '-ffp-contract=off',
                   str(source), '-o', str(library)]
        result = subprocess.run(command, text=True, capture_output=True)
        (directory / ('build_' + digest[:16] + '.log')).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        (directory / ('build_' + digest[:16] + '.json')).write_text(json.dumps({
            'source_sha256': digest, 'command': command,
            'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}, indent=2))
    loaded = ctypes.CDLL(str(library))
    fn = loaded.coordiworld_controller
    fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
    fn.restype = ctypes.c_int
    fn._library = loaded
    _FUNCTION = fn
    return fn


def simulate_arrays(proposals, initial, sampling, tracker, motion_model):
    from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import get_velocity_curvature_profiles_with_derivatives_from_poses
    proposals = np.ascontiguousarray(proposals[:, :sampling.num_poses + 1], np.float64)
    steps = sampling.num_poses
    dt = sampling.interval_length
    velocity, _, curvature, _ = get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=dt, poses=proposals[..., :3], jerk_penalty=tracker._jerk_penalty,
        curvature_rate_penalty=tracker._curvature_rate_penalty)
    assert len(proposals.shape) == 3 and proposals.shape[1:] == (steps + 1, 11)
    parameters = np.array([dt, tracker._wheel_base, motion_model._vehicle.wheel_base,
        motion_model._accel_time_constant, motion_model._steering_angle_time_constant,
        motion_model._max_steering_angle, tracker._q_longitudinal, tracker._r_longitudinal,
        *np.diag(tracker._q_lateral), tracker._r_lateral[0, 0],
        tracker._stopping_proportional_gain, tracker._stopping_velocity], np.float64)
    arrays = [proposals, np.ascontiguousarray(velocity), np.ascontiguousarray(curvature),
              np.ascontiguousarray(initial, np.float64), parameters]
    assert all(np.isfinite(a).all() for a in arrays)
    output = np.empty_like(proposals)
    rc = kernel()(len(proposals), steps, tracker._tracking_horizon,
                  *[a.ctypes.data for a in arrays], output.ctypes.data)
    if rc:
        raise RuntimeError(f'compiled controller failed: {rc}')
    return output


class CompiledPDMSimulator(StructuredPDMSimulator):
    def simulate_proposals(self, states, initial_ego_state):
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self.proposal_sampling.interval_length
        return simulate_arrays(states, ego_state_to_state_array(initial_ego_state),
            self.proposal_sampling, self._tracker, self._motion_model)
