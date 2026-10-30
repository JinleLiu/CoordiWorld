"""Typed paired XY, heading, and instantaneous velocity from the frozen solver."""
import torch
from .swept_occupancy_graph_timestep import CapturedSweptRail
from .swept_occupancy_fused_timestep import rollout


class CapturedSweptFullState(CapturedSweptRail):
    def _forward(self):
        states,_=rollout(self.paths,self.info,self.ego,self.initial,
                         iterations=self.iterations,pieces=self.pieces)
        delta=states-states[self.reference:self.reference+1]
        heading=torch.atan2(delta[...,2].sin(),delta[...,2].cos())
        return torch.cat((delta[...,:2],heading[...,None],delta[...,3:5],
                          torch.zeros_like(delta[...,5:7])),-1)
