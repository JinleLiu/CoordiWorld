"""Current/history-only motion prior plus structured learned acceleration.

Development revision; no new calibration/existence/causal qualification implied.
The base structured history/map encoder and recurrent interaction blocks remain.
"""
import torch
from torch import nn
from ..model import CoordiWorld, mlp
from ..geometry import wrap_angle


def motion_history(x):
    cur, old = x.agents[:, -1], x.agents[:, -2]
    speed = torch.linalg.vector_norm(cur[..., 3:5], dim=-1)
    moving = x.agent_valid[:, -1] & (x.agent_class == 1) & (speed > .5)
    history = moving & x.agent_valid[:, -2] & (torch.linalg.vector_norm(old[..., 3:5], dim=-1) > .5)
    yaw = torch.atan2(cur[..., 4], cur[..., 3])
    previous = torch.atan2(old[..., 4], old[..., 3])
    direction = torch.stack((yaw.cos(), yaw.sin()), -1)
    acceleration = (((cur[..., 3:5] - old[..., 3:5]) * direction).sum(-1) / .5).clamp(-4, 3)
    turn = (wrap_angle(yaw - previous) / .5).clamp(-.5, .5)
    return cur, speed, yaw, torch.where(history, acceleration, 0), torch.where(history, turn, 0), moving, history


def kinematic_prior(x, horizon=None, tau=2.):
    cur, speed, yaw, acceleration, turn, moving, history = motion_history(x)
    horizon = horizon or x.candidates.shape[2]
    pos = cur[..., :2]; states = []
    for k in range(horizon):
        decay = torch.exp(cur.new_tensor(-((k + .5) * .5) / tau))
        ns = (speed + acceleration * decay * .5).clamp_min(0)
        ny = yaw + turn * decay * .5
        mid = (yaw + ny) / 2
        pos = pos + ((speed + ns) * .25)[..., None] * torch.stack((mid.cos(), mid.sin()), -1)
        velocity = ns[..., None] * torch.stack((ny.cos(), ny.sin()), -1)
        state = torch.cat((pos, wrap_angle(ny)[..., None], velocity, cur[..., 5:7]), -1)
        cv = torch.cat((cur[..., :2] + (k + 1) * .5 * cur[..., 3:5], cur[..., 2:]), -1)
        states.append(torch.where(moving[..., None], state, cv))
        speed, yaw = ns, ny
    return torch.stack(states, 1)


class KinematicResidualCoordiWorld(CoordiWorld):
    """Moving-vehicle decoder; other entities follow the declared CV control."""
    def __init__(self, config, tau=2.):
        super().__init__(config)
        if config.action_conditioned or config.dt != .5:
            raise ValueError('This factual revision uses an action-blind 0.5s protocol')
        self.tau = tau
        self.motion_feedback = mlp(10, config.hidden_dim, config.hidden_dim)
        self.acceleration_head = mlp(config.hidden_dim + 10, config.hidden_dim, 2)
        nn.init.zeros_(self.acceleration_head[-1].weight)
        nn.init.zeros_(self.acceleration_head[-1].bias)

    def rollout(self, x, encoded=None):
        b, m, h, _ = x.candidates.shape
        if m != 1: raise ValueError('Factual evaluation must remove candidate actions')
        context, maps = self.encode(x) if encoded is None else encoded
        cur, speed, yaw, acceleration, turn, moving, history = motion_history(x)
        initial_yaw = yaw; initial_speed = speed; pos = cur[..., :2]
        hidden = context; states = []
        for k in range(h):
            decay = torch.exp(cur.new_tensor(-((k + .5) * .5) / self.tau))
            features = torch.stack((initial_speed / 20, speed / 20, acceleration / 4, turn / .5,
                (yaw-initial_yaw).sin(), (yaw-initial_yaw).cos(), history.to(cur.dtype),
                torch.full_like(speed, (k + 1) / h), cur[..., 5] / 10, cur[..., 6] / 10), -1)
            hidden = hidden + .1 * self.motion_feedback(features)
            for block in self.recurrent:
                hidden = block(hidden, hidden[:, :1], maps, x.agent_valid[:, -1], x.map_valid, False)
            residual = torch.tanh(self.acceleration_head(torch.cat((hidden, features), -1)))
            ns = (speed + (acceleration * decay + 2 * residual[..., 0]) * .5).clamp_min(0)
            ny = yaw + (turn * decay + .2 * residual[..., 1]) * .5
            mid = (yaw + ny) / 2
            pos = pos + ((speed + ns) * .25)[..., None] * torch.stack((mid.cos(), mid.sin()), -1)
            velocity = ns[..., None] * torch.stack((ny.cos(), ny.sin()), -1)
            state = torch.cat((pos, wrap_angle(ny)[..., None], velocity, cur[..., 5:7]), -1)
            cv = torch.cat((cur[..., :2] + (k + 1) * .5 * cur[..., 3:5], cur[..., 2:]), -1)
            states.append(torch.where(moving[..., None], state, cv)); speed, yaw = ns, ny
        states = torch.stack(states, 1)[:, None]
        valid = x.agent_valid[:, -1, None, None].expand(b, 1, h, -1)
        return {'states': states, 'existence': valid.to(states.dtype) * torch.sigmoid(states.new_tensor(3.)),
                'variance': states.new_full((*states.shape[:-1], 2), .74314718056)}
