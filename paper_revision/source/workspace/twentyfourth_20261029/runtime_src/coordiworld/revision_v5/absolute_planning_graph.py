"""Absolute states of the unchanged current-input physical rollout."""
from .sparse_swept_graph import CapturedSparseSweptFullState
from .swept_full_state_graph import CapturedSweptFullState
from .sparse_swept_rollout import rollout as sparse_rollout
from .swept_occupancy_fused_timestep import rollout as dense_rollout


class CapturedAbsoluteSparse(CapturedSparseSweptFullState):
    def _forward(self):
        states, _ = sparse_rollout(self.paths, self.info, self.ego, self.initial,
                                  self.neighbors, self.neighbor_valid,
                                  iterations=self.iterations, pieces=self.pieces)
        return states


class CapturedAbsoluteDense(CapturedSweptFullState):
    def _forward(self):
        states, _ = dense_rollout(self.paths, self.info, self.ego, self.initial,
                                 iterations=self.iterations, pieces=self.pieces)
        return states
