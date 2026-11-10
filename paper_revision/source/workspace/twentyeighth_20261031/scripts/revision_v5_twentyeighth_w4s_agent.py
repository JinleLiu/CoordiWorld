"""Observational wrapper: native official W4S inference and controls unchanged."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from world4scorer_agent import World4ScorerAgent,B2D_ORDER
def get_entry_point():return 'R18World4ScorerAgent'
class R18World4ScorerAgent(World4ScorerAgent):
 def setup(self,path):
  self.r18cfg=json.load(open(path.split('+')[0]));t=time.perf_counter();super().setup(path);self.setup_seconds=time.perf_counter()-t
  self.out=Path(self.r18cfg['output_dir']);self.out.mkdir(parents=True,exist_ok=True);self.trace=open(self.out/'trace.jsonl','w',buffering=1);self.replans=0;self._last_outputs={}
  def observed(mod,args,out):
   self._last_outputs={k:v.detach().float().cpu().numpy()for k,v in out.items()if torch.is_tensor(v)and v.numel()<1000000};self._last_ego=args[0]['ego_status'].detach().cpu().numpy()
  self.hook=self.model.register_forward_hook(observed)
  meta=dict(method='World4Scorer_route_tp',identity='official-pretrained',official_commit='7814310e0bd65c6e21f06b81eedfec2ac9a06aa9',checkpoint_sha256=hashlib.sha256(Path(self.r18cfg['ckpt']).read_bytes()).hexdigest(),config=self.r18cfg,torch=torch.__version__,setup_seconds=self.setup_seconds,input_permissions='fourRGB; actual hero pose/velocity/acceleration; dense navigation route; no surrounding actor access',native_rules='official route bonus4.0, command retention,10tick replan, PID and creep unmodified',wrapper='logging and forward observation only',sensors=self.sensors())
  (self.out/'agent_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 def _replan(self,data,timestamp,pose):
  begin=time.perf_counter();super()._replan(data,timestamp,pose);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
  np.savez_compressed(self.out/f'replan_{self.replans:05d}.npz',**self._last_outputs,ego_status=self._last_ego,plan_world=self.plan,pose=pose,timestamp=timestamp,seconds=elapsed)
  if self.replans==0:np.savez_compressed(self.out/'first_sensor_input.npz',**{k:data[k][1]for k in B2D_ORDER})
  self.replans+=1;self.last_plan_seconds=elapsed
 def run_step(self,data,timestamp):
  t=time.perf_counter();ctrl=super().run_step(data,timestamp);v=self.hero_actor.get_velocity();p=self._pose();self.trace.write(json.dumps(dict(step=self.tick,sim_seconds=float(timestamp),pose=p.tolist(),velocity_world=[v.x,-v.y,v.z],control=dict(steer=ctrl.steer,throttle=ctrl.throttle,brake=ctrl.brake),run_step_seconds=time.perf_counter()-t,replans=self.replans,plan_seconds=getattr(self,'last_plan_seconds',None),creep_left=self.creep_left,creep_cool=self.creep_cool))+'\n');return ctrl
 def destroy(self):
  if hasattr(self,'hook'):self.hook.remove()
  if hasattr(self,'trace'):self.trace.close()
  super().destroy()
