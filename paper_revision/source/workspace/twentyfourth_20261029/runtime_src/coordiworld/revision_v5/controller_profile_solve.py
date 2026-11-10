"""Same regularized profile objective, direct full-rank normal-equation solve.

Development candidate only. Different floating point arithmetic is tested
against the pinned pseudoinverse implementation before use in a world model.
No actor, candidate or output cache; zero-speed curvature keeps native penalty.
"""
import numpy as np

def solve_profile(a,y,regularizer):
 at=a.transpose(0,2,1)
 gram=np.einsum('bij,bjk->bik',at,a)+regularizer
 rhs=np.einsum('bij,bj->bi',at,y)
 try:
  x=np.linalg.solve(gram,rhs[...,None])[...,0]
 except np.linalg.LinAlgError:
  return np.einsum('bij,bj->bi',np.einsum('bij,bjk->bik',np.linalg.pinv(gram),at),y)
 residual=np.max(np.abs(np.einsum('bij,bj->bi',gram,x)-rhs),axis=1)
 scale=np.max(np.abs(gram),axis=(1,2))*np.max(np.abs(x),axis=1)*gram.shape[-1]+np.max(np.abs(rhs),axis=1)
 bad=~np.isfinite(x).all(1)|(residual>1e-12*np.maximum(scale,1e-12))
 if bad.any():x[bad]=np.einsum('bij,bj->bi',np.einsum('bij,bjk->bik',np.linalg.pinv(gram[bad]),at[bad]),y[bad])
 return x

def get_velocity_curvature_profiles_with_derivatives_from_poses(discretization_time,poses,jerk_penalty,curvature_rate_penalty):
 from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import _get_xy_heading_displacements_from_poses,_make_banded_difference_matrix,_generate_profile_from_initial_condition_and_derivatives,INITIAL_CURVATURE_PENALTY
 assert discretization_time>0 and jerk_penalty>0 and curvature_rate_penalty>0
 dt=discretization_time;xy,dh=_get_xy_heading_displacements_from_poses(poses);b,n=dh.shape;y=xy.reshape(b,-1);h=np.asarray(poses[:,:-1,2],np.float64)
 column=np.zeros_like(y);column[:,0::2]=np.cos(h);column[:,1::2]=np.sin(h)
 a=np.repeat(column[...,None]*dt**2,n,axis=2);a[...,0]=column*dt;mask=np.repeat(np.triu(np.ones((n,n),bool),k=1),2,axis=0);a[:,mask]=0.
 band=_make_banded_difference_matrix(n-2);r=np.block([np.zeros((len(band),1)),band]);r=np.repeat(r[None],b,axis=0);reg=jerk_penalty*np.einsum('bij,bjk->bik',r.transpose(0,2,1),r)
 x=solve_profile(a,y,reg);acc=x[:,1:];velocity=_generate_profile_from_initial_condition_and_derivatives(x[:,0],acc,dt)
 a=np.repeat(np.tri(n,dtype=np.float64)[None],b,axis=0);a[:,:,0]=velocity*dt;a[:,1:,1:]*=(velocity*dt**2)[:,None,1:].transpose(0,2,1);q=curvature_rate_penalty*np.eye(n);q[0,0]=INITIAL_CURVATURE_PENALTY
 x=solve_profile(a,dh,q);rate=x[:,1:];curvature=_generate_profile_from_initial_condition_and_derivatives(x[:,0],rate,dt)
 return velocity,acc,curvature,rate
