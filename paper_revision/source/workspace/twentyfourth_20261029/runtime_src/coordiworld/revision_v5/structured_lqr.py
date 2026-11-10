"""Exact algebra of the pinned triangular LQR transition; float64 reductions.

Each lateral transition is [[1,a,0],[0,1,b],[0,0,1]], with an
affine heading increment c and constant steering-rate input dt.
Prefix sums replace ten dense matrix products, without changing the
linearized dynamics, costs, horizon, clipping, or bicycle rollout.
"""
import numpy as np
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def prefix_before(values):
    return np.concatenate((np.zeros_like(values[:,:1]),np.cumsum(values[:,:-1],axis=1)),axis=1)


class StructuredLQRTracker(BatchLQRTracker):
    def _lateral_lqr_controller(self,initial_lateral_state_vector,velocity_profile,curvature_profile):
        assert velocity_profile.shape[-1]==self._tracking_horizon
        assert curvature_profile.shape==velocity_profile.shape
        count=velocity_profile.shape[0];steps=self._tracking_horizon;dt=self._discretization_time
        a=velocity_profile*dt;b=a/self._wheel_base
        # Preserve the native multiplication order for the affine term.
        c=-velocity_profile*curvature_profile*dt
        A=np.broadcast_to(np.eye(3,dtype=np.float64),(count,3,3)).copy()
        A[:,0,1]=a.sum(1);A[:,1,2]=b.sum(1);A[:,0,2]=(a*prefix_before(b)).sum(1)
        B=np.zeros((count,3,1),np.float64)
        previous_steering=dt*np.arange(steps,dtype=np.float64)
        B[:,2,0]=steps*dt;B[:,1,0]=(b*previous_steering).sum(1)
        B[:,0,0]=(a*prefix_before(b*previous_steering)).sum(1)
        g=np.zeros((count,3),np.float64);g[:,1]=c.sum(1);g[:,0]=(a*prefix_before(c)).sum(1)
        command=self._solve_one_step_lateral_lqr(initial_lateral_state_vector,A,B,g)
        return np.squeeze(command,axis=-1)


class StructuredPDMSimulator(PDMSimulator):
    def __init__(self,proposal_sampling):
        super().__init__(proposal_sampling)
        self._tracker=StructuredLQRTracker()
