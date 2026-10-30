"""Tensor contracts. Model inputs never contain logged future states or labels."""
from dataclasses import dataclass, fields, replace
import torch
from torch import Tensor


@dataclass
class SceneInputs:
    # Coordinates are in the CURRENT ego rear-axle frame, x forward/y left.
    # State channels: x, y, yaw, vx, vy, length, width (SI units).
    ego: Tensor                 # [B,L,7]
    agents: Tensor              # [B,L,N,7]
    agent_valid: Tensor         # [B,L,N], observed, not predicted existence
    agent_class: Tensor         # [B,N], 0 unknown, 1 vehicle, 2 pedestrian, 3 bicycle...
    confidence: Tensor          # [B,L,N,3], class/track/localization
    confidence_valid: Tensor    # [B,L,N,3], unknown != confidence zero
    source: Tensor              # [B,L,N,6] multi-hot: null/record/map/control/infra/generic
    lineage: Tensor             # [B,L,N], 0 null, 1 original record, 2 fused
    ambiguity: Tensor           # [B,L,N], current observation conflict flag
    map_features: Tensor        # [B,P,8], x,y,heading,length,width,signal,signal_valid,age
    map_valid: Tensor           # [B,P]
    map_kind: Tensor            # [B,P], 0 null, 1 lane, 2 drivable, 3 stop, 4 signal
    map_source: Tensor          # [B,P,6], same multi-hot convention
    candidates: Tensor          # [B,M,H,3], x,y,yaw (rear axle)
    candidate_valid: Tensor     # [B,M], same for all compared methods
    rule_margins: Tensor        # [B,M,H,P,4], input-map geometry, NOT future annotations
    rule_valid: Tensor          # [B,M,H,P,4]
    ego_rear_axle_to_center: Tensor  # [B], explicit geometry convention

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def index(self, index):
        return replace(self, **{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def clone(self):
        return replace(self, **{f.name: getattr(self, f.name).clone() for f in fields(self)})

    def select_candidates(self, indices):
        # One selected candidate per scene; retain M dimension.
        batch = torch.arange(self.ego.shape[0], device=self.ego.device)
        return replace(self,
            candidates=self.candidates[batch, indices][:, None],
            candidate_valid=self.candidate_valid[batch, indices][:, None],
            rule_margins=self.rule_margins[batch, indices][:, None],
            rule_valid=self.rule_valid[batch, indices][:, None])

    def validate(self):
        b, l, n, d = self.agents.shape
        m, h = self.candidates.shape[1:3]
        p = self.map_features.shape[1]
        assert d == 7 and self.ego.shape == (b,l,7)
        assert self.agent_valid.shape == (b,l,n)
        assert self.confidence.shape == self.confidence_valid.shape == (b,l,n,3)
        assert self.source.shape == (b,l,n,6) and self.map_source.shape == (b,p,6)
        assert self.rule_margins.shape == self.rule_valid.shape == (b,m,h,p,4)
        assert self.candidate_valid.any(-1).all(), 'Every scene needs at least one valid candidate'
        for f in fields(self):
            x = getattr(self, f.name)
            assert x.shape[0] == b, f.name
            if x.is_floating_point(): assert torch.isfinite(x).all(), f.name
        assert (self.agents[...,5:7][self.agent_valid] > 0).all()
        assert ((self.confidence[self.confidence_valid] >= 0) & (self.confidence[self.confidence_valid] <= 1)).all()
        return self


@dataclass
class Supervision:
    future_agents: Tensor       # [B,H,N,7], logged observations only
    future_valid: Tensor        # [B,H,N]
    logged_ego: Tensor          # [B,H,3], used ONLY as Stage-I action input
    logged_rule_margins: Tensor # [B,1,H,P,4], current map evaluated on logged ego
    logged_rule_valid: Tensor   # [B,1,H,P,4]
    collision: Tensor          # [B,M], geometric replay labels or official evaluator labels
    violation: Tensor          # [B,M]
    violation_valid: Tensor    # [B,M], missing maps are not negative labels
    utility: Tensor            # [B,M], always name the metric/provider in metadata
    progress: Tensor           # [B,M]
    collision_partners: Tensor # [B,M,N]
    # Optional official scorer labels affect pairwise preference only. Keep the
    # logged geometric labels separate for calibrated risk/partner diagnostics.
    rank_collision: Tensor | None = None
    rank_violation: Tensor | None = None
    rank_utility: Tensor | None = None
    rank_progress: Tensor | None = None
    rank_valid: Tensor | None = None

    def to(self, device):
        return replace(self, **{f.name:getattr(self,f.name).to(device) if getattr(self,f.name) is not None else None for f in fields(self)})

    def index(self, index):
        return replace(self, **{f.name:getattr(self,f.name)[index] if getattr(self,f.name) is not None else None for f in fields(self)})


def cat_inputs(items):
    return SceneInputs(**{f.name:torch.cat([getattr(x,f.name) for x in items]) for f in fields(SceneInputs)})


def cat_targets(items):
    values={}
    for field in fields(Supervision):
        group=[getattr(x,field.name) for x in items]
        if all(v is None for v in group):values[field.name]=None
        elif any(v is None for v in group):raise ValueError('Mixed ranking label providers within a batch')
        else:values[field.name]=torch.cat(group)
    return Supervision(**values)
