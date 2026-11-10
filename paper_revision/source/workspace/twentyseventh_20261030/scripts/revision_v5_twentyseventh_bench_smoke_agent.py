"""Official-stack infrastructure smoke, NOT a CoordiWorld policy result.

Uses CARLA's packaged BasicAgent and logs current structured observations.
No frozen PNC head is invoked. A successful route validates the simulator,
official runner and logger, not CoordiWorld's closed-loop performance.
"""
import os,json,time,math,re
from pathlib import Path
import numpy as np
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent,Track
from agents.navigation.basic_agent import BasicAgent

def get_entry_point():return 'InfrastructureSmokeAgent'

def xyz(v):return [float(v.x),float(v.y),float(v.z)]
def pose(t):return dict(location=xyz(t.location),rotation_degrees=[float(t.rotation.roll),float(t.rotation.pitch),float(t.rotation.yaw)])

class InfrastructureSmokeAgent(AutonomousAgent):
 def set_global_plan(self,gps,world):
  super().set_global_plan(gps,world);self._dense_world_plan=world
 def setup(self,path_to_conf_file):
  config_path,_,self.route_context=path_to_conf_file.partition('+')
  self.track=Track.MAP;self.cfg=json.load(open(config_path));self.out=Path(self.cfg['output_dir'])/re.sub('[^a-zA-Z0-9_-]','_',self.route_context or 'route');self.out.mkdir(parents=True,exist_ok=True)
  self.driver=BasicAgent(self.hero_actor,target_speed=float(self.cfg.get('target_speed_kmh',25.0)))
  world=self.hero_actor.get_world();self.map=world.get_map();plan=[(self.map.get_waypoint(t.location),o)for t,o in self._dense_world_plan];self.driver.set_global_plan(plan)
  self.step=0;self.start_wall=time.monotonic();self.trace=open(self.out/'trace.jsonl','w',buffering=1);self.structured=open(self.out/'current_structured_frames.jsonl','w',buffering=1)
  pc=self.hero_actor.get_physics_control()
  meta=dict(role='INFRASTRUCTURE_SMOKE_ONLY_NOT_COORDIWORLD',agent='CARLA packaged BasicAgent',coordiworld_models_used=False,track='MAP',privileged_current_actor_access=True,route_waypoints=[pose(t)|dict(road_option=str(o))for t,o in self._dense_world_plan],town=self.map.name,hero_type=self.hero_actor.type_id,hero_bbox_extent=xyz(self.hero_actor.bounding_box.extent),vehicle_mass=pc.mass,wheels=[dict(position_cm=xyz(w.position),max_steer_angle=w.max_steer_angle,radius_cm=w.radius)for w in pc.wheels],config=self.cfg)
  (self.out/'agent_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 def sensors(self):return [dict(type='sensor.speedometer',reading_frequency=20,id='speed'),dict(type='sensor.other.imu',x=0.,y=0.,z=0.,roll=0.,pitch=0.,yaw=0.,sensor_tick=.05,id='imu')]
 def run_step(self,input_data,timestamp):
  begin=time.perf_counter();control=self.driver.run_step();ego=self.hero_actor;t=ego.get_transform();v=ego.get_velocity();acc=ego.get_acceleration();wp=self.map.get_waypoint(t.location)
  row=dict(step=self.step,sim_seconds=float(timestamp),wall_elapsed=time.monotonic()-self.start_wall,ego=pose(t),velocity_world=xyz(v),acceleration_world=xyz(acc),angular_velocity_degrees_s=xyz(ego.get_angular_velocity()),speed_mps=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),control=dict(steer=control.steer,throttle=control.throttle,brake=control.brake),road_id=wp.road_id,lane_id=wp.lane_id,lane_width_m=wp.lane_width,current_speed_limit_kmh=ego.get_speed_limit(),at_traffic_light=ego.is_at_traffic_light(),traffic_light_state=str(ego.get_traffic_light_state()),basic_agent_step_seconds=time.perf_counter()-begin)
  self.trace.write(json.dumps(row)+'\n')
  if self.step%10==0:
   actors=[]
   for actor in ego.get_world().get_actors():
    if actor.id==ego.id or not(actor.type_id.startswith('vehicle.')or actor.type_id.startswith('walker.pedestrian.')):continue
    at=actor.get_transform()
    if at.location.distance(t.location)>100:continue
    actors.append(dict(id=actor.id,type=actor.type_id,pose=pose(at),velocity_world=xyz(actor.get_velocity()),acceleration_world=xyz(actor.get_acceleration()),box_extent=xyz(actor.bounding_box.extent),box_offset=xyz(actor.bounding_box.location)))
   self.structured.write(json.dumps(dict(step=self.step,sim_seconds=float(timestamp),coordinate='CARLA left-handed world meters, rotations degrees; raw, not silently relabeled as NAVSIM',current_only=True,source_radius_m=100.,ego=row,actors=actors))+'\n')
  self.step+=1;return control
 def destroy(self):
  if hasattr(self,'trace'):
   self.trace.close();self.structured.close();(self.out/'agent_end.json').write_text(json.dumps(dict(agent_steps=self.step,wall_seconds=time.monotonic()-self.start_wall,coordiworld_models_used=False))+'\n')
