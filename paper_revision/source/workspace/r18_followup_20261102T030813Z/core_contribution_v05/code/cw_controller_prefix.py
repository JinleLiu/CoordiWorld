"""0.5s controller-matched bicycle prefix, with live CARLA vehicle parameters.

The PID and low-speed release are the actual inherited controller functions.
The plant is a deterministic approximate bicycle/torque model, NOT CARLA future
ground truth. Prefix error must be measured on each development episode before
confirmation is permitted. This model does not replace the real controller.
"""
import copy,math
from types import SimpleNamespace,MethodType
import numpy as np
from cw_followup_physics import rotation

VERSION='controller_prefix_v04_signed_longitudinal_velocity'
CONFIG=dict(dt=.05,horizon_s=.5,max_position_error_p95_m=.50,max_heading_error_p95_rad=.20,
            drivetrain_efficiency=.75,rolling_coefficient=.01,steer_normalized_rate_per_s=5.,
            max_acceleration_mps2=6.,max_braking_mps2=8.,validation_required=True)

def physics_receipt(hero,geometry):
    p=hero.get_physics_control();w=p.wheels
    result=dict(source='current CARLA actor.get_physics_control, shared all internal arms',
       mass_kg=float(p.mass),drag_coefficient=float(p.drag_coefficient),max_rpm=float(p.max_rpm),
       final_ratio=float(p.final_ratio),torque_curve=[[float(v.x),float(v.y)] for v in p.torque_curve],
       steering_curve=[[float(v.x),float(v.y)] for v in p.steering_curve],
       forward_gears=[float(g.ratio) for g in p.forward_gears],
       gear_switch_time_s=float(p.gear_switch_time),clutch_strength=float(p.clutch_strength),
       use_gear_autobox=bool(p.use_gear_autobox),
       wheels=[dict(max_steer_angle_deg=float(x.max_steer_angle),radius_m=float(x.radius)/100.,max_brake_torque_nm=float(x.max_brake_torque),tire_friction=float(x.tire_friction)) for x in w],
       wheelbase_m=float(geometry['wheelbase']),front_track_m=float(abs(geometry['wheels_body_m'][0][1]-geometry['wheels_body_m'][1][1])))
    assert result['mass_kg']>100 and result['wheelbase_m']>1 and max(x['max_steer_angle_deg'] for x in result['wheels'])>0
    return result

def snapshot(agent):
    c=agent.hero_actor.get_control()
    import carla
    angles=[agent.hero_actor.get_wheel_steer_angle(x) for x in [carla.VehicleWheelLocation.FL_Wheel,carla.VehicleWheelLocation.FR_Wheel]]
    phi=-np.radians(angles);ys=-np.asarray(agent.geometry['wheels_body_m'])[:2,1];wb=agent.geometry['wheelbase'];curvature=np.tan(phi)/(wb+ys*np.tan(phi))
    return dict(turn_history=list(agent.turn.win),speed_history=list(agent.speedpid.win),
      actual_control=dict(steer=float(c.steer),throttle=float(c.throttle),brake=float(c.brake),gear=int(c.gear),reverse=bool(c.reverse),manual_gear_shift=bool(c.manual_gear_shift)),
      measured_front_steer_rad=float(np.arctan(wb*curvature.mean())),actual_wheel_steer_deg=[float(v) for v in angles],
      inherited_tracker='revision_v5_twentyeighth_cw_agent.tracker plus low_speed.release')

def prepare_tracker(frame,candidate,native,state):
    from revision_v5_twentyeighth_cw_agent import PID,CoordiWorldAgent
    yaw=float(frame['body_pose_math'][2]);rear=np.asarray(frame['ego2global_translation'][:2]);off=np.asarray(frame['rear_geometry']['rear_offset_math'])
    ca=np.asarray(candidate);ra=np.r_[np.r_[rear,yaw][None],np.c_[ca[:,:2]@rotation(yaw).T+rear,ca[:,2]+yaw]]
    plan=ra.copy();plan[:,:2]-=rotation(ra[:,2])@off
    obj=SimpleNamespace(plan=plan,native=native,plan_t0=0.,cfg=dict(speed_source='native_preview_v02',speed_preview_seconds=1.),turn=PID(1.25,.75,.3),speedpid=PID(5.,.5,1.))
    obj.turn.win.extend(state['turn_history']);obj.speedpid.win.extend(state['speed_history']);obj.at=MethodType(CoordiWorldAgent.at,obj)
    return obj

def control(obj,t,body,body_velocity):
    from revision_v5_twentyeighth_cw_agent import CoordiWorldAgent
    from revision_v5_twentyeighth_low_speed import release
    v=SimpleNamespace(x=float(body_velocity[0]),y=float(-body_velocity[1]),z=0.)
    obj.hero_actor=SimpleNamespace(get_velocity=lambda:v)
    c=CoordiWorldAgent.tracker(obj,t,body)
    release(c,obj.desired,math.hypot(v.x,v.y),obj.speedpid)
    return c

def roll(frame,candidate,native,state,physics,assume_engaged=False):
    obj=prepare_tracker(frame,candidate,native,state)
    off=np.asarray(frame['rear_geometry']['rear_offset_math']);origin=np.asarray(frame['ego2global_translation'][:2]);yaw0=float(frame['body_pose_math'][2]);x,y=origin;yaw=yaw0
    vx,vy,ax,ay=np.asarray(frame['ego_dynamic_state'],float);v=float(vx);delta=float(state['measured_front_steer_rad'])
    max_delta=math.radians(max(w['max_steer_angle_deg'] for w in physics['wheels']));wheelbase=physics['wheelbase_m'];radius=float(np.mean([w['radius_m'] for w in physics['wheels']]));mass=physics['mass_kg']
    torque=np.asarray(physics['torque_curve']);steercurve=np.asarray(physics['steering_curve']);gears=np.asarray(physics['forward_gears']);gears=gears[gears>0]
    assert len(gears) and radius>0
    observed_gear=int(state.get('actual_control',{}).get('gear',1));hold_neutral=observed_gear==0 and not assume_engaged
    paths=[[0.,0.,0.]];commands=[];states=[];omega_prev=float(frame['omega'])
    for tick in range(round(CONFIG['horizon_s']/CONFIG['dt'])):
        body=np.r_[np.array([x,y])-rotation(yaw)@off,yaw]
        # Original tracker reads body velocity, while the policy state is rear
        # axle velocity. Undo the yaw-rate lever arm before calling that PID.
        velocity=rotation(yaw)@np.array([v+omega_prev*off[1],vy-omega_prev*off[0]])
        c=control(obj,tick*CONFIG['dt'],body,velocity)
        steering_scale=float(np.interp(abs(v)*3.6,steercurve[:,0],steercurve[:,1]));inner=-c.steer*max_delta*steering_scale
        # The live stationary API fixture shows inner/outer Ackermann angles:
        # steer=-.2 yields -14.0/-12.349 degrees, not two equal14deg wheels.
        target=math.atan(wheelbase*math.tan(inner)/(wheelbase+physics['front_track_m']/2*abs(math.tan(inner))))
        delta+=np.clip(target-delta,-CONFIG['steer_normalized_rate_per_s']*max_delta*CONFIG['dt'],CONFIG['steer_normalized_rate_per_s']*max_delta*CONFIG['dt'])
        rpm_by_gear=max(abs(v),.5)/radius*physics['final_ratio']*gears*60/(2*np.pi)
        suitable=np.flatnonzero(rpm_by_gear<=physics['max_rpm']*.8);g=min(observed_gear-1,len(gears)-1) if observed_gear>0 else (int(suitable[0]) if len(suitable) else len(gears)-1)
        rpm=max(750.,rpm_by_gear[g]);engine=float(np.interp(rpm,torque[:,0],torque[:,1]));drive=c.throttle*engine*gears[g]*physics['final_ratio']*CONFIG['drivetrain_efficiency']/(mass*radius)
        if hold_neutral:drive=0.
        brake=c.brake*min(CONFIG['max_braking_mps2'],sum(w['max_brake_torque_nm']/w['radius_m'] for w in physics['wheels'])/mass)
        extent=frame['rear_geometry']['box_extent'];area=4*float(extent[1])*float(extent[2]);drag=.5*1.225*physics['drag_coefficient']*area*v*v/mass
        # Keep the measured signed rear velocity. Clipping rollback to zero
        # changes the real PID's speed error at t=0. Resistance opposes motion;
        # forward-moving integration remains exactly the prior computation.
        if v>=0.:
            acc=float(np.clip(drive-brake-drag-(CONFIG['rolling_coefficient']*9.81 if v>0 else 0),-CONFIG['max_braking_mps2'],CONFIG['max_acceleration_mps2']))
            vnew=max(0.,v+acc*CONFIG['dt'])
        else:
            acc=float(np.clip(drive+brake+drag+CONFIG['rolling_coefficient']*9.81,-CONFIG['max_braking_mps2'],CONFIG['max_acceleration_mps2']))
            vnew=v+acc*CONFIG['dt']
            if vnew>0.:
                # Braking cannot itself launch the vehicle forwards after it
                # arrests rollback. Only residual forward drive can do that.
                remaining=max(0.,CONFIG['dt']+v/acc)
                vnew=max(0.,min(CONFIG['max_acceleration_mps2'],drive-brake-CONFIG['rolling_coefficient']*9.81))*remaining
        vmid=(v+vnew)/2;omega=vmid*math.tan(delta)/wheelbase;ymid=yaw+omega*CONFIG['dt']/2
        x+=CONFIG['dt']*(vmid*math.cos(ymid)-vy*math.sin(ymid));y+=CONFIG['dt']*(vmid*math.sin(ymid)+vy*math.cos(ymid));yaw+=omega*CONFIG['dt'];v=vnew;vy*=.9;omega_prev=omega
        p=(np.array([x,y])-origin)@rotation(yaw0);paths.append([*p,math.atan2(math.sin(yaw-yaw0),math.cos(yaw-yaw0))]);commands.append([float(c.steer),float(c.throttle),float(c.brake)]);states.append(dict(v=v,acc=acc,yaw_rate=omega,front_steer_rad=delta))
    return np.asarray(paths),dict(version=VERSION,plant='approximate bicycle/torque; measured-prefix validation required',observed_gear=observed_gear,gear_assumption='current neutral held through prefix; an engaged alternative is also checked' if hold_neutral else 'observed forward gear' if observed_gear>0 else 'immediate forward engagement alternative',config=CONFIG,commands=commands,states=states)
