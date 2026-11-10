"""Batched official W4 PID/creep + published CARLA Garage motion predictor.

Math XY/yaw view is the reflection of CARLA XY/yaw. No observations after t0.
Motion coefficients/formulas: carla_garage commit48e6a6d, team_code/config.py
and kinematic_bicycle_model.py. W4 tracking: commit7814310e, world4scorer_agent.py.
The physical approximation is a research executor, not actual future execution.
"""
import numpy as np
FRONT=-.090769015;REAR=1.4178275;GAIN=.36848336;DT=.05
THROTTLE=np.array([9.63873001e-1,4.37535692e-4,-3.80192912e-1,1.74950069,9.16787414e-2,-7.05461530e-2,-1.05996152e-3,6.71079346e-4])
BRAKE=np.array([9.31711370e-3,8.20967431e-2,-2.83832427e-3,5.06587474e-5,-4.90357228e-7,2.44419284e-9,-4.91381935e-12])
def rotate(h,z):
 z=np.asarray(z);return np.stack([np.cos(h)*z[...,0]-np.sin(h)*z[...,1],np.sin(h)*z[...,0]+np.cos(h)*z[...,1]],-1)
def rotate_back(h,z):return rotate(-h,z)
def snapshot(agent,body_speed):
 return dict(turn_win=list(agent.turn_pid.win),speed_win=list(agent.speed_pid.win),still_ticks=agent.still_ticks,creep_left=agent.creep_left,creep_cool=agent.creep_cool,body_speed=float(body_speed))
def rollout(body,state,offset,controller,steps=80,return_controls=False):
 body=np.asarray(body,dtype=np.float64);n=len(body);off=np.asarray(offset);p0=np.asarray(state['body_pose']);p=np.repeat(p0[None],n,0);v=np.full(n,controller['body_speed']);t0=controller.get('plan_t0_seconds',state['timestamp_us']/1e6)
 plan=np.zeros((n,9,2));plan[:,0]=p0[:2];plan[:,1:]=p0[:2]+rotate(p0[2],body[...,:2])
 def at(times):
  times=np.minimum(np.maximum(np.asarray(times),0),3.99);ix=np.minimum((times/.5).astype(int),7);f=(times-ix*.5)/.5
  return plan[:,ix]+(plan[:,ix+1]-plan[:,ix])*f[...,None]
 tw=np.tile(np.asarray(controller['turn_win']),(n,1));sw=np.tile(np.asarray(controller['speed_win']),(n,1));assert tw.shape==(n,40)and sw.shape==(n,40)
 still=np.full(n,controller['still_ticks'],int);left=np.full(n,controller['creep_left'],int);cool=np.full(n,controller['creep_cool'],int)
 # Fixed NAVSIM StateIndex: steering_rate8, angular_velocity9, angular_acceleration10.
 states=np.zeros((n,steps+1,11));states[:,0,:3]=state['rear_pose'];states[:,0,3:5]=state['rear_velocity_body'];states[:,0,5:7]=state['rear_acceleration_body'];states[:,0,7]=state.get('tire_angle_surrogate',0);states[:,0,9]=state['yaw_rate'];states[:,0,10]=state['yaw_acceleration']
 last_vworld=rotate(p[:,2],np.tile(state['rear_velocity_body'],(n,1)));last_omega=np.full(n,state['yaw_rate']);last_wheel=states[:,0,7].copy();controls=[]
 for k in range(steps):
  tau=(t0+k*DT)-t0;p_now=at(tau);p_next=at(tau+.5);dp=rotate_back(p[:,2],p_next-p_now);desired=np.maximum(0,dp[:,0])*2
  aimd=np.where(v>6.5,10.,4.);times=[];tt=tau
  while tt<3.99:times.append(tt);tt+=.05
  if times:
   ahead=at(times);hit=np.linalg.norm(ahead-p[:,None,:2],axis=2)>=aimd[:,None];first=hit.argmax(1);aim=ahead[np.arange(n),first];aim=np.where(hit.any(1)[:,None],aim,at(3.99))
  else:aim=at(3.99)
  delta_xy=rotate_back(p[:,2],aim-p[:,:2]);angle=np.where(delta_xy[:,0]<.5,0,np.degrees(np.arctan2(delta_xy[:,1],delta_xy[:,0]))/90.)
  tw=np.roll(tw,-1,axis=1);tw[:,-1]=angle;steer=np.clip(-(1.25*angle+.75*tw.mean(1)+.3*(tw[:,-1]-tw[:,-2])),-1,1)
  brake=(desired<.15)|(v/np.maximum(desired,.1)>1.1);delta=np.clip(desired-v,0,.25);sw=np.roll(sw,-1,axis=1);sw[:,-1]=delta;throttle=np.clip(5*delta+.5*sw.mean(1)+(sw[:,-1]-sw[:,-2]),0,.75);throttle=np.where(brake,0,throttle)
  still=np.where(v<.5,still+1,0);active=left>0;cooling=(~active)&(cool>0);launch=(~active)&(~cooling)&(still>240);throttle=np.where(active,.4,throttle);brake=brake&(~active);left=np.where(active,left-1,left);cool=np.where(active&(left==0),40,cool);cool=np.where(cooling,cool-1,cool);left=np.where(launch,60,left)
  # CARLA VehicleControl exposes binary32 fields; mirror this wire rounding.
  u=np.stack([steer,throttle,brake],1).astype(np.float32).astype(np.float64);controls.append(u)
  wheel=GAIN*u[:,0];beta=np.arctan(REAR/(FRONT+REAR)*np.tan(wheel));omega=-v/REAR*np.sin(beta)
  p[:,:2]+=np.stack([v*np.cos(p[:,2]-beta),v*np.sin(p[:,2]-beta)],1)*DT;p[:,2]+=omega*DT
  vk=v*3.6;th=u[:,1];features=np.stack([vk,vk**2,th,th**2,vk*th,vk*th**2,vk**2*th,vk**2*th**2],1)
  power=vk[:,None]**np.arange(1,8);vb=power@BRAKE/3.6;vt=features@THROTTLE/3.6;v=np.maximum(0,np.where(u[:,2]>0,vb,np.where(th<.3,v,vt)))
  omega_new=-v/REAR*np.sin(beta);vbody=np.stack([v*np.cos(beta)-omega_new*off[1],-v*np.sin(beta)+omega_new*off[0]],1);vw=rotate(p[:,2],vbody);ab=rotate_back(p[:,2],(vw-last_vworld)/DT)
  st=states[:,k+1];st[:,:2]=p[:,:2]+rotate(p[:,2],off);st[:,2]=p[:,2];st[:,3:5]=vbody;st[:,5:7]=ab;st[:,7]=-wheel;st[:,8]=(-wheel-last_wheel)/DT;st[:,9]=omega_new;st[:,10]=(omega_new-last_omega)/DT;last_vworld=vw;last_omega=omega_new;last_wheel=-wheel
 assert np.isfinite(states).all()
 return (states,np.stack(controls,1))if return_controls else states
