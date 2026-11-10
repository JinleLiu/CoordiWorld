"""Release only the inherited low-speed brake floor; no creep/minimum throttle."""
import numpy as np
FLOOR=.02  # m/s, replaces .15 only; chosen plan's requested velocity is retained.
def release(ctrl,desired,speed,pid):
 if not(FLOOR<=desired<.15)or speed/max(desired,.1)>1.1:return False
 # tracker already advanced this PID exactly once; reuse, never step twice.
 value=pid.kp*pid.win[-1]+pid.ki*np.mean(pid.win)+pid.kd*(pid.win[-1]-pid.win[-2])
 ctrl.throttle=float(np.clip(value,0,.75));ctrl.brake=0.;return True
