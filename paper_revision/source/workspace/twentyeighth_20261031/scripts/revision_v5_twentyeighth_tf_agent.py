"""Observational wrapper of pinned native TF++, one preregistered checkpoint."""
import json,time,hashlib
from pathlib import Path
import torch,numpy as np
from sensor_agent import SensorAgent
def get_entry_point():return 'R18TransfuserAgent'
class R18TransfuserAgent(SensorAgent):
 def set_global_plan(self,gps,world):self.r18_dense=list(world);super().set_global_plan(gps,world)
 def setup(self,path,route_index=None,traffic_manager=None):
  cfgpath,_,suffix=path.partition('+');self.r18cfg=json.load(open(cfgpath));t=time.perf_counter();super().setup(self.r18cfg['checkpoint_dir']+('+'+suffix if suffix else''),route_index,traffic_manager);self.r18out=Path(self.r18cfg['output_dir']);self.r18out.mkdir(parents=True,exist_ok=True);self.r18trace=open(self.r18out/'trace.jsonl','w',buffering=1);self.r18steps=0
  route=[[tr.location.x,tr.location.y,tr.location.z,tr.rotation.roll,tr.rotation.pitch,tr.rotation.yaw,str(cmd)]for tr,cmd in self.r18_dense]
  r=dict(method='TFpp_all_towns_seed0',identity='official-pretrained-native-single-checkpoint',official_commit='48e6a6dbcb133220333bd65e242fbc2b492bb15b',native_model_count=len(self.nets),config=self.r18cfg,setup_seconds=time.perf_counter()-t,torch=torch.__version__,numpy=np.__version__,sensors=self.sensors(),route_world=route,route_sha=hashlib.sha256(json.dumps(route).encode()).hexdigest(),permissions='native RGB LiDAR GNSS IMU speed and navigation route; wrapper reads actual pose for logging only; it never feeds actual pose to inference',wrapper='logging only; native controller, action repeat, UKF, creep and safety logic preserved')
  assert len(self.nets)==1;(self.r18out/'agent_metadata.json').write_text(json.dumps(r,indent=2)+'\n')
 def run_step(self,data,timestamp,sensors=None):
  t=time.perf_counter();c=super().run_step(data,timestamp,sensors);torch.cuda.synchronize();elapsed=time.perf_counter()-t
  from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
  actor=CarlaDataProvider.get_hero_actor();tr=actor.get_transform();v=actor.get_velocity();self.r18trace.write(json.dumps(dict(step=self.step,sim_seconds=float(timestamp),ego_world=dict(x=tr.location.x,y=tr.location.y,z=tr.location.z,yaw=tr.rotation.yaw),velocity_world=[v.x,v.y,v.z],control=dict(steer=c.steer,throttle=c.throttle,brake=c.brake),run_step_seconds=elapsed,initialized=self.initialized))+'\n');self.r18steps+=1;return c
 def destroy(self):
  if hasattr(self,'r18trace'):self.r18trace.close()
  super().destroy()
