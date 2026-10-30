"""Preserve body orientation when the motion decoder integrates velocity yaw.

The observed initial body/motion offset is held fixed; this is a kinematic
approximation, not a learned slip-angle model. XY and velocity stay identical.
"""
import torch
from ..geometry import wrap_angle
from .kinematic_residual import KinematicResidualCoordiWorld


def body_heading_states(scene, states):
    cur=scene.agents[:,-1]
    moving=scene.agent_valid[:,-1] & (scene.agent_class==1) & (torch.linalg.vector_norm(cur[...,3:5],dim=-1)>.5)
    offset=wrap_angle(cur[...,2]-torch.atan2(cur[...,4],cur[...,3]))
    out=states.clone();previous_xy=cur[:,None,:,:2];previous_yaw=cur[:,None,:,2]
    for k in range(states.shape[2]):
        proposed=wrap_angle(states[:,:,k,:,2]+offset[:,None])
        translates=torch.linalg.vector_norm(states[:,:,k,:,:2]-previous_xy,dim=-1)>1e-7
        yaw=torch.where(translates,proposed,previous_yaw)
        out[:,:,k,:,2]=torch.where(moving[:,None],yaw,states[:,:,k,:,2])
        previous_xy=states[:,:,k,:,:2];previous_yaw=out[:,:,k,:,2]
    return out


class BodyHeadingKinematicResidual(KinematicResidualCoordiWorld):
    def rollout(self,x,encoded=None):
        out=super().rollout(x,encoded);out['states']=body_heading_states(x,out['states']);return out
