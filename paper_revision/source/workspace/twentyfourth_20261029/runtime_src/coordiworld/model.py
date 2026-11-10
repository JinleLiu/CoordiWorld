"""History encoding -> recurrent agent rollout -> geometric risk -> ranking."""
from dataclasses import dataclass, field
import torch
from torch import nn
from .schema import SceneInputs
from .geometry import rear_axle_to_box, sat_clearance, wrap_angle


@dataclass
class ModelConfig:
    hidden_dim:int=128
    heads:int=4
    history_layers:int=2
    feedforward_dim:int=512
    dropout:float=0.1
    rollout_blocks:int=1
    dt:float=0.5
    max_history:int=8
    max_agents:int=32
    use_provenance:bool=True
    use_confidence:bool=True
    action_conditioned:bool=True
    collision_aggregation:str='noisy_or'
    uncertainty_aggregation:str='existence_mean'
    motion_prior:str='constant_velocity'
    position_residual_scale:float=2.0
    velocity_residual_scale:float=2.0
    yaw_residual_scale:float=0.3
    rule_temperatures:list=field(default_factory=lambda:[0.5]*4)
    violation_beta:float=8.0
    score_weights:list=field(default_factory=lambda:[1.0,1.0,0.1])
    uncertainty_scale:float=25.0


def mlp(input_dim,hidden,output):
    return nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Linear(hidden,output))


def map_risk(x,temperatures,beta):
    available=x.rule_valid & x.map_valid[:,None,None,:,None]
    soft=torch.sigmoid(-x.rule_margins/temperatures)
    flat=soft.flatten(2);mask=available.flatten(2)
    weights=torch.softmax((beta*flat).masked_fill(~mask,-1e4),-1)*mask
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    return (weights*flat).sum(-1),mask.any(-1)


def normalized_state(x):
    return torch.cat((x[...,:2]/50,torch.sin(x[...,2:3]),torch.cos(x[...,2:3]),x[...,3:5]/20,x[...,5:7]/10),-1)


def safe_attention(module,query,key,key_valid):
    # Always-valid null key prevents NaNs for scenes with no agents/map tokens.
    null=torch.zeros((key.shape[0],1,key.shape[-1]),device=key.device,dtype=key.dtype)
    key=torch.cat((key,null),1)
    padding=torch.cat((~key_valid,torch.zeros((key.shape[0],1),device=key.device,dtype=torch.bool)),1)
    return module(query,key,key,key_padding_mask=padding,need_weights=False)[0]


class InteractionBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg.hidden_dim
        self.action=nn.MultiheadAttention(d,cfg.heads,dropout=cfg.dropout,batch_first=True)
        self.agents=nn.MultiheadAttention(d,cfg.heads,dropout=cfg.dropout,batch_first=True)
        self.maps=nn.MultiheadAttention(d,cfg.heads,dropout=cfg.dropout,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.update=mlp(d,d,d);self.gate=nn.Linear(d*2,d)

    def forward(self,h,actions,maps,agent_valid,map_valid,conditioned=True):
        original=h
        if conditioned:h=self.norms[0](h+self.action(h,actions,actions,need_weights=False)[0])
        h=self.norms[1](h+safe_attention(self.agents,h,h,agent_valid))
        h=self.norms[2](h+safe_attention(self.maps,h,maps,map_valid))
        gate=torch.sigmoid(self.gate(torch.cat((original,h),-1)))
        return (original+gate*self.update(h))*agent_valid[...,None]


class CoordiWorld(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig();c=self.config;d=c.hidden_dim
        self.state_embedding=mlp(8,d,d)
        self.map_embedding=mlp(8,d,d)
        self.class_embedding=nn.Embedding(16,d)
        self.slot_embedding=nn.Embedding(c.max_agents,d)
        self.source_embedding=nn.Embedding(6,d)
        self.lineage_embedding=nn.Embedding(3,d)
        self.ambiguity_embedding=nn.Embedding(2,d)
        self.map_kind_embedding=nn.Embedding(5,d)
        self.time_embedding=nn.Embedding(c.max_history,d)
        self.type_embedding=nn.Embedding(3,d)
        self.confidence_embedding=mlp(6,d,d)
        self.confidence_null=nn.Parameter(torch.zeros(3,d))
        self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,c.heads,c.feedforward_dim,c.dropout,batch_first=True,norm_first=True),c.history_layers,enable_nested_tensor=False)
        self.action_embedding=mlp(5,d,d)
        self.recurrent=nn.ModuleList([InteractionBlock(c) for _ in range(c.rollout_blocks)])
        self.motion_head=mlp(d,d,8)
        self.collision_head=mlp(11,d,1)
        # A constant-velocity residual prior starts near physical motion.
        nn.init.zeros_(self.motion_head[-1].weight);nn.init.zeros_(self.motion_head[-1].bias)
        self.motion_head[-1].bias.data[5]=3.0
        nn.init.zeros_(self.collision_head[-1].weight);nn.init.zeros_(self.collision_head[-1].bias)
        self.register_buffer('uncertainty_scale',torch.tensor(c.uncertainty_scale))
        self.register_buffer('score_weights',torch.tensor(c.score_weights))
        self.register_buffer('rule_temperatures',torch.tensor(c.rule_temperatures))
        self.register_buffer('source_counts',torch.zeros(6,dtype=torch.long))

    def encode(self,x:SceneInputs):
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
        map_scale=x.map_features.new_tensor([50,50,3.14159265,50,10,1,1,5])
        maps=self.map_embedding(x.map_features/map_scale)+self.map_kind_embedding(x.map_kind)+self.type_embedding.weight[2]
        if self.config.use_provenance:maps=maps+x.map_source.to(maps.dtype) @ self.source_embedding.weight.to(maps.dtype)
        tokens=torch.cat((dynamic.reshape(b,l*(n+1),d),maps),1)
        valid=torch.cat((dynamic_valid.flatten(1),x.map_valid),1)
        tokens=self.history(tokens,src_key_padding_mask=~valid)
        return tokens[:,:l*(n+1)].reshape(b,l,n+1,d)[:,-1,1:],tokens[:,l*(n+1):]

    def rollout(self,x,encoded=None):
        b,m,h,_=x.candidates.shape;n=x.agents.shape[2];p=x.map_features.shape[1];d=self.config.hidden_dim
        agents,maps=self.encode(x) if encoded is None else encoded
        hidden=agents[:,None].expand(-1,m,-1,-1).reshape(b*m,n,d)
        maps=maps[:,None].expand(-1,m,-1,-1).reshape(b*m,p,d)
        valid=x.agent_valid[:,-1,None,:].expand(-1,m,-1).reshape(b*m,n)
        map_valid=x.map_valid[:,None,:].expand(-1,m,-1).reshape(b*m,p)
        candidate=x.candidates.reshape(b*m,h,3)
        time=torch.arange(1,h+1,device=candidate.device,dtype=candidate.dtype)*self.config.dt
        action_features=torch.cat((candidate[...,:2]/50,candidate[...,2:3].sin(),candidate[...,2:3].cos(),time[None,:,None].expand(b*m,-1,-1)/4),-1)
        action=self.action_embedding(action_features)
        current=x.agents[:,-1,None].expand(-1,m,-1,-1).reshape(b*m,n,7)
        states=[];existence=[];variances=[]
        for k in range(h):
            for block in self.recurrent:
                hidden=block(hidden,action[:,k:k+1],maps,valid,map_valid,self.config.action_conditioned)
            output=self.motion_head(hidden).float()
            dx=torch.tanh(output[...,:2])*self.config.position_residual_scale
            if self.config.motion_prior=='constant_velocity':dx=dx+current[...,3:5]*self.config.dt
            elif self.config.motion_prior!='none':raise ValueError(self.config.motion_prior)
            position=current[...,:2]+dx
            yaw=wrap_angle(current[...,2:3]+torch.tanh(output[...,2:3])*self.config.yaw_residual_scale)
            velocity=current[...,3:5]+torch.tanh(output[...,3:5])*self.config.velocity_residual_scale
            current=torch.cat((position,yaw,velocity,current[...,5:7]),-1)
            states.append(current.reshape(b,m,n,7))
            existence.append((torch.sigmoid(output[...,5])*valid).reshape(b,m,n))
            # Variance is positive, bounded for stable NLL, and explicitly position variance.
            variances.append((torch.nn.functional.softplus(output[...,6:8])+0.05).clamp_max(2500).reshape(b,m,n,2))
        return {'states':torch.stack(states,2),'existence':torch.stack(existence,2),'variance':torch.stack(variances,2)}

    def evaluate_risk(self,x,rollout):
        c=self.config;states=rollout['states'];variance=rollout['variance'];exist=rollout['existence']
        ego_box=rear_axle_to_box(x.candidates,x.ego[:,-1,5:7][:,None,None],x.ego_rear_axle_to_center[:,None,None])
        agent_box=torch.cat((states[...,:3],states[...,5:7]),-1)
        gap=sat_clearance(ego_box[:,:,:,None],agent_box)
        delta=states[...,:2]-ego_box[:,:,:,None,:2]
        yaw=x.candidates[...,2,None];co,si=yaw.cos(),yaw.sin()
        longitudinal=delta[...,0]*co+delta[...,1]*si
        lateral=-delta[...,0]*si+delta[...,1]*co
        ego_velocity=torch.diff(torch.cat((torch.zeros_like(x.candidates[:,:,:1,:2]),x.candidates[...,:2]),2),dim=2)/c.dt
        relative_velocity=states[...,3:5]-ego_velocity[:,:,:,None]
        closing=-(delta*relative_velocity).sum(-1)/delta.square().sum(-1).sqrt().clamp_min(0.1)
        relative_yaw=states[...,2]-yaw
        interactions=torch.stack((delta[...,0]/50,delta[...,1]/50,longitudinal/20,lateral/10,gap/10,relative_yaw.sin(),relative_yaw.cos(),closing/20,variance[...,0].log()/5,variance[...,1].log()/5,exist),-1)
        # Prevent a random 0.5 probability per event from saturating noisy-OR at
        # initialization. This geometric prior remains trainable via a residual.
        prior=-gap/(variance.sum(-1).sqrt()+0.5)-2.0
        logits=self.collision_head(interactions.to(next(self.collision_head.parameters()).dtype)).squeeze(-1).float()+prior
        instant=torch.sigmoid(logits)*exist
        if c.collision_aggregation=='noisy_or':collision=-torch.expm1(torch.log1p(-instant.clamp(max=1-1e-6)).sum((-1,-2)))
        elif c.collision_aggregation=='max':collision=instant.amax((-1,-2))
        else:raise ValueError(c.collision_aggregation)
        violation,violation_available=map_risk(x,self.rule_temperatures,c.violation_beta)
        uncertainty_weights=exist
        if c.uncertainty_aggregation=='relevance_weighted':uncertainty_weights=exist*torch.exp(-gap.clamp_min(0)/5)
        elif c.uncertainty_aggregation!='existence_mean':raise ValueError(c.uncertainty_aggregation)
        uncertainty=(variance.sum(-1)*uncertainty_weights).sum((-1,-2))/uncertainty_weights.sum((-1,-2)).clamp_min(1e-8)
        normalized=(uncertainty/self.uncertainty_scale.clamp_min(1e-6)).clamp(0,1)
        score=self.score_weights[0]*collision+self.score_weights[1]*violation+self.score_weights[2]*normalized
        return {'collision':collision,'violation':violation,'violation_available':violation_available,'uncertainty':uncertainty,'normalized_uncertainty':normalized,'score':score,'instant_collision':instant,'clearance':gap}

    def forward(self,x):
        rollout=self.rollout(x)
        return {**rollout,**self.evaluate_risk(x,rollout)}

    def select(self,x,output):
        return output['score'].masked_fill(~x.candidate_valid,float('inf')).argmin(-1)
