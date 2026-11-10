"""Execution-only graph replay of a frozen factual model, with fresh inputs."""
from dataclasses import fields
import torch
from ..model import normalized_state
from ..geometry import wrap_angle
from .kinematic_residual import motion_history
from .body_heading import body_heading_states


def input_signature(scene):
    return tuple((f.name,tuple(getattr(scene,f.name).shape),str(getattr(scene,f.name).dtype)) for f in fields(scene))


class CapturedFactual:
    def __init__(self,model,scene):
        self.model=model.eval().requires_grad_(False)
        self.inputs=scene.clone();self.signature=input_signature(scene)
        self.map_scale=scene.map_features.new_tensor([50,50,3.14159265,50,10,1,1,5])
        self.decays=[torch.exp(scene.agents.new_tensor(-((k+.5)*.5)/model.tau)) for k in range(scene.candidates.shape[2])]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):self._forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph),torch.inference_mode():self.output=self._forward()
        torch.cuda.synchronize()

    def _forward(self):
        return graph_states(self.model,self.inputs,self.map_scale,self.decays)

    @torch.inference_mode()
    def __call__(self,scene):
        if input_signature(scene)!=self.signature:raise ValueError('Factual graph input signature changed')
        for f in fields(scene):getattr(self.inputs,f.name).copy_(getattr(scene,f.name))
        self.graph.replay()
        return self.output.clone()

def graph_encode(self,x,map_scale):
    b,l,n,_=x.agents.shape;d=self.config.hidden_dim
    if l>self.config.max_history:raise ValueError('History exceeds configured maximum')
    agents=self.state_embedding(normalized_state(x.agents))+self.class_embedding(x.agent_class)[:,None]
    agents=agents+self.slot_embedding(torch.arange(n,device=x.ego.device))[None,None]
    if self.config.use_confidence:
        values=torch.where(x.confidence_valid,x.confidence,torch.zeros_like(x.confidence))
        agents=agents+self.confidence_embedding(torch.cat((values,x.confidence_valid.float()),-1))
        agents=agents+(~x.confidence_valid).to(agents.dtype) @ self.confidence_null.to(agents.dtype)
    if self.config.use_provenance:
        agents=agents+x.source.to(agents.dtype) @ self.source_embedding.weight.to(agents.dtype)+self.lineage_embedding(x.lineage)+self.ambiguity_embedding(x.ambiguity.long())
    agents=agents+self.type_embedding.weight[1]
    ego=self.state_embedding(normalized_state(x.ego))+self.type_embedding.weight[0]
    temporal=self.time_embedding(torch.arange(l,device=x.ego.device))
    dynamic=torch.cat((ego[:,:,None],agents),2)+temporal[None,:,None]
    dynamic_valid=torch.cat((torch.ones((b,l,1),device=x.ego.device,dtype=torch.bool),x.agent_valid),-1)
    maps=self.map_embedding(x.map_features/map_scale)+self.map_kind_embedding(x.map_kind)+self.type_embedding.weight[2]
    if self.config.use_provenance:maps=maps+x.map_source.to(maps.dtype) @ self.source_embedding.weight.to(maps.dtype)
    tokens=torch.cat((dynamic.reshape(b,l*(n+1),d),maps),1)
    valid=torch.cat((dynamic_valid.flatten(1),x.map_valid),1)
    tokens=self.history(tokens,src_key_padding_mask=~valid)
    return tokens[:,:l*(n+1)].reshape(b,l,n+1,d)[:,-1,1:],tokens[:,l*(n+1):]


def graph_states(self, x, map_scale, decays):
    b, m, h, _ = x.candidates.shape
    if m != 1: raise ValueError('Factual evaluation must remove candidate actions')
    context, maps = graph_encode(self,x,map_scale)
    cur, speed, yaw, acceleration, turn, moving, history = motion_history(x)
    initial_yaw = yaw; initial_speed = speed; pos = cur[..., :2]
    hidden = context; states = []
    for k in range(h):
        decay = decays[k]
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
    return body_heading_states(x,states)[0,0]
