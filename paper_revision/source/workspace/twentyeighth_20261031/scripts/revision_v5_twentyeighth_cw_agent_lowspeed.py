"""v04 exploratory actual tracker: execute small nonzero requested speeds."""
import json,math,hashlib
from pathlib import Path
from revision_v5_twentyeighth_cw_agent_shortpath import CoordiWorldShortPathAgent
import revision_v5_twentyeighth_low_speed as low
def get_entry_point():return 'CoordiWorldLowSpeedAgent'
class CoordiWorldLowSpeedAgent(CoordiWorldShortPathAgent):
 def setup(self,path):
  super().setup(path);self.low_speed_releases=0
  p=self.out/'agent_metadata.json';m=json.load(open(p));m['low_speed_tracking']=dict(version='v04_exploratory',brake_floor_mps=low.FLOOR,original_floor_mps=.15,source_sha256=hashlib.sha256(Path(low.__file__).read_bytes()).hexdigest(),change='release low-speed brake only; reuse same once-updated PID; no minimum throttle, velocity floor, creep or model change');p.write_text(json.dumps(m,indent=2)+'\n')
 def tracker(self,timestamp,body):
  ctrl=super().tracker(timestamp,body)
  if self.plan is not None:
   v=self.hero_actor.get_velocity();changed=low.release(ctrl,self.desired,math.hypot(v.x,v.y),self.speedpid);self.low_speed_releases+=int(changed)
  return ctrl
 def destroy(self):
  if hasattr(self,'out'):(self.out/'low_speed_stats.json').write_text(json.dumps(dict(released_calls=getattr(self,'low_speed_releases',0)),indent=2)+'\n')
  super().destroy()
