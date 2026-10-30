"""OpenScene log adapter matching NAVSIM's rear-axle coordinate convention."""
from dataclasses import asdict
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..schema import SceneInputs,Supervision,cat_inputs,cat_targets
from ..candidates import CandidateConfig,generate_candidates
from ..geometry import wrap_angle,collision_labels,map_rule_margins


CLASSES={'vehicle':1,'pedestrian':2,'bicycle':3,'traffic_cone':4,'barrier':5,'czone_sign':6,'generic_object':7}
PACIFICA_LENGTH=5.176
PACIFICA_WIDTH=2.297
PACIFICA_REAR_AXLE_TO_CENTER=1.461


class LogUnpickler(pickle.Unpickler):
    """Only numerical arrays and builtins needed by verified official logs."""
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.numeric','_frombuffer')}
        if (module,name) in allowed:return super().find_class(module,name)
        if module=='builtins' and name in ['set','frozenset','slice','complex']:return super().find_class(module,name)
        raise pickle.UnpicklingError(f'Unexpected class in data: {module}.{name}')


def load_log(path):
    with open(path,'rb') as stream:return LogUnpickler(stream).load()


def ego_pose(frame):
    # R14: exact fixed NAVSIM Scene/AgentInput quaternion convention.
    from pyquaternion import Quaternion
    yaw=Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]
    return np.r_[np.asarray(frame['ego2global_translation'],float)[:2],yaw]


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def transform_pose(pose,origin):
    return np.r_[rotation(-origin[2]) @ (pose[:2]-origin[:2]),wrap_angle(pose[2]-origin[2])]


def frame_agents(frame,origin):
    pose=ego_pose(frame);relative_yaw=wrap_angle(pose[2]-origin[2]);rotate=rotation(relative_yaw)
    translation=rotation(-origin[2]) @ (pose[:2]-origin[:2])
    annotations=frame['anns'];out={}
    for box,velocity,name,track in zip(annotations['gt_boxes'],annotations['gt_velocity_3d'],annotations['gt_names'],annotations['track_tokens']):
        box=np.asarray(box,float);velocity=np.asarray(velocity,float)
        if not np.isfinite(box).all() or not np.isfinite(velocity[:2]).all() or min(box[3:5])<=0:continue
        state=np.r_[rotate@box[:2]+translation,wrap_angle(box[6]+relative_yaw),rotate@velocity[:2],box[3:5]]
        out[str(track)]=(state,CLASSES.get(str(name),0))
    return out


def make_inputs(history,config,map_tokens=None,candidate_bundle=None):
    """Inference adapter: this API accepts historical/current frames only."""
    dc=config['data'];l=dc['history_frames'];h=dc['future_steps'];n=dc['max_agents'];p=dc['max_map_tokens'];dt=dc['dt']
    if len(history)!=l:raise ValueError('Incorrect history length')
    current=history[-1];origin=ego_pose(current)
    all_times=np.array([f['timestamp'] for f in history],dtype=np.float64)/1e6
    if not np.allclose(np.diff(all_times),dt,atol=0.08):raise ValueError('Non-contiguous timestamps')
    now_agents=frame_agents(current,origin)
    eligible=[(np.linalg.norm(state[:2]),track) for track,(state,_) in now_agents.items() if np.linalg.norm(state[:2])<=dc['roi_m']]
    selected=[track for _,track in sorted(eligible)[:n]]
    agents=np.zeros((l,n,7),np.float32);valid=np.zeros((l,n),bool);classes=np.zeros(n,np.int64)
    ego=np.zeros((l,7),np.float32)
    for t,frame in enumerate(history):
        pose=ego_pose(frame);relative=transform_pose(pose,origin)
        velocity=rotation(pose[2]-origin[2]) @ np.asarray(frame['ego_dynamic_state'][:2])
        ego[t]=np.r_[relative,velocity,PACIFICA_LENGTH,PACIFICA_WIDTH]
        available=frame_agents(frame,origin)
        for j,track in enumerate(selected):
            if track in available:agents[t,j],classes[j]=available[track];valid[t,j]=True
    confidence=np.zeros((l,n,3),np.float32);confidence_valid=np.zeros_like(confidence,bool)
    source=np.zeros((l,n,6),np.float32);source[...,1]=valid;source[...,0]=~valid
    lineage=valid.astype(np.int64)
    speed=float(np.linalg.norm(ego[-1,3:5]));yaw_rate=float(wrap_angle(ego[-1,2]-ego[-2,2])/dt)
    candidates,candidate_valid,manifest=candidate_bundle if candidate_bundle is not None else generate_candidates(speed,yaw_rate,h,dt,CandidateConfig(**config['candidates']))
    map_tokens=(map_tokens or [])[:p]
    features=np.zeros((p,8),np.float32);map_valid=np.zeros(p,bool);kind=np.zeros(p,np.int64);map_source=np.zeros((p,6),np.float32);map_source[:,0]=1
    for j,token in enumerate(map_tokens):
        features[j]=token['features'];map_valid[j]=token.get('available',True)
        kind[j]={'lane':1,'drivable':2,'stop':3,'signal':4}[token['kind']]
        map_source[j]=0;map_source[j,3 if kind[j]==4 else 2]=1
    m=len(candidates)
    margins=np.zeros((m,h,p,4),np.float32);rule_valid=np.zeros_like(margins,bool)
    if map_tokens:
        part,mask=map_rule_margins(candidates,[PACIFICA_LENGTH,PACIFICA_WIDTH],PACIFICA_REAR_AXLE_TO_CENTER,map_tokens)
        margins[:,:,:len(map_tokens)]=part;rule_valid[:,:,:len(map_tokens)]=mask
    def tensor(value):return torch.from_numpy(np.asarray(value))[None]
    inputs=SceneInputs(ego=tensor(ego),agents=tensor(agents),agent_valid=tensor(valid),agent_class=tensor(classes),confidence=tensor(confidence),confidence_valid=tensor(confidence_valid),source=tensor(source),lineage=tensor(lineage),ambiguity=tensor(np.zeros((l,n),bool)),map_features=tensor(features),map_valid=tensor(map_valid),map_kind=tensor(kind),map_source=tensor(map_source),candidates=tensor(candidates),candidate_valid=tensor(candidate_valid),rule_margins=tensor(margins),rule_valid=tensor(rule_valid),ego_rear_axle_to_center=torch.tensor([PACIFICA_REAR_AXLE_TO_CENTER]))
    meta={'scene_id':str(current['token']),'log_name':str(current['log_name']),'map_name':str(current['map_location']),'timestamp':int(current['timestamp']),'tracks':selected,'eligible_agents':len(eligible),'retained_agents':len(selected),'truncated_agents':max(0,len(eligible)-n),'map_tokens':len(map_tokens),'candidate_manifest':manifest,'confidence_source':'unknown; all confidence_valid flags false','input_latest_timestamp':int(current['timestamp'])}
    meta['map_token_metadata']=[{k:t[k] for k in ['id','kind','semantic_scope','rule_enabled','signal_valid','prohibited','initially_stopped'] if k in t} for t in map_tokens]
    return inputs.validate(),meta


def make_sample(frames,current_index,config,map_tokens=None,candidate_bundle=None):
    """Attach supervision AFTER independently constructing inference inputs."""
    dc=config['data'];l=dc['history_frames'];h=dc['future_steps'];n=dc['max_agents'];p=dc['max_map_tokens'];dt=dc['dt']
    history=frames[current_index-l+1:current_index+1];future=frames[current_index+1:current_index+1+h]
    if len(history)!=l or len(future)!=h:raise ValueError('Insufficient history/future')
    times=np.array([f['timestamp'] for f in history+future],float)/1e6
    if not np.allclose(np.diff(times),dt,atol=.08):raise ValueError('Non-contiguous timestamps')
    inputs,meta=make_inputs(history,config,map_tokens,candidate_bundle)
    origin=ego_pose(history[-1]);selected=meta['tracks'];map_tokens=(map_tokens or [])[:p]
    def tensor(value):return torch.from_numpy(np.asarray(value))[None]
    future_agents=np.zeros((h,n,7),np.float32);future_valid=np.zeros((h,n),bool);logged_ego=[]
    for k,frame in enumerate(future):
        available=frame_agents(frame,origin);logged_ego.append(transform_pose(ego_pose(frame),origin))
        for j,track in enumerate(selected):
            if track in available:future_agents[k,j]=available[track][0];future_valid[k,j]=True
    logged_ego=np.asarray(logged_ego,np.float32)
    logged_margins=np.zeros((1,h,p,4),np.float32);logged_rule_valid=np.zeros_like(logged_margins,bool)
    if map_tokens:
        part,mask=map_rule_margins(logged_ego[None],[PACIFICA_LENGTH,PACIFICA_WIDTH],PACIFICA_REAR_AXLE_TO_CENTER,map_tokens)
        logged_margins[:,:,:len(map_tokens)]=part;logged_rule_valid[:,:,:len(map_tokens)]=mask
    collision,partners,_=collision_labels(inputs,tensor(future_agents),tensor(future_valid))
    violation=torch.any((inputs.rule_margins<0)&inputs.rule_valid,dim=(-1,-2,-3)).float()
    available_rule=inputs.rule_valid.flatten(2).any(-1)
    # Development label only: progress along the CURRENT heading, not EPDMS.
    speed=float(inputs.ego[0,-1,3:5].norm())
    progress=inputs.candidates[:,:,-1,0].clamp_min(0)/max(speed*h*dt,1.0)
    targets=Supervision(future_agents=tensor(future_agents),future_valid=tensor(future_valid),logged_ego=tensor(logged_ego),logged_rule_margins=tensor(logged_margins),logged_rule_valid=tensor(logged_rule_valid),collision=collision,violation=violation,violation_valid=available_rule,utility=progress.clone(),progress=progress,collision_partners=partners)
    meta.update({'label_provider':'logged_box_replay_plus_heading_progress_v1','official_epdms':False})
    return inputs.validate(),targets,meta


def split_for_log(log_name,ratios=(.7,.1,.1,.1),seed=0):
    value=int(hashlib.sha256(f'{seed}:{log_name}'.encode()).hexdigest()[:16],16)/2**64
    return ['train','validation','calibration','test'][min(int(np.searchsorted(np.cumsum(ratios),value,side='right')),3)]


def log_sampling_key(log_name,seed=0):
    # A separate hash domain prevents prefix sampling from selecting only train.
    return hashlib.sha256(f'log-sampling:{seed}:{log_name}'.encode()).hexdigest()


def save_sample(path,inputs,targets,metadata):
    arrays={'input_'+k:v.cpu().numpy() for k,v in vars(inputs).items()}
    arrays.update({'target_'+k:v.cpu().numpy() for k,v in vars(targets).items() if v is not None})
    arrays['metadata_json']=np.asarray(json.dumps(metadata,sort_keys=True))
    np.savez_compressed(path,**arrays)


class CachedScenes(Dataset):
    def __init__(self,manifest,split):
        self.root=Path(manifest).parent
        data=json.loads(Path(manifest).read_text())
        self.records=[r for r in data['scenes'] if r['split']==split]
        self.metadata=data
        if not self.records:raise ValueError(f'No scenes for split {split}')

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index]
        with np.load(self.root/record['path'],allow_pickle=False) as data:
            x=SceneInputs(**{k.removeprefix('input_'):torch.from_numpy(data[k].copy()) for k in data.files if k.startswith('input_')})
            y=Supervision(**{k.removeprefix('target_'):torch.from_numpy(data[k].copy()) for k in data.files if k.startswith('target_')})
            meta=json.loads(str(data['metadata_json']))
        return x,y,meta


def collate_scenes(items):
    return cat_inputs([x[0] for x in items]),cat_targets([x[1] for x in items]),[x[2] for x in items]
