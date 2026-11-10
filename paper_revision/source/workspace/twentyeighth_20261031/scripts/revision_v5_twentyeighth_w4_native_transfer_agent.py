"""Separate frozen research variant; leaves the original transfer agent intact."""
import json,math,subprocess
from revision_v5_twentyeighth_w4_transfer_agent import R18W4TransferAgent,B,B2D_ORDER
from revision_v5_twentyeighth_w4_execution import send_packet
from revision_v5_twentyeighth_native_rollout import snapshot
def get_entry_point():return 'R18NativeTransferAgent'
class R18NativeTransferAgent(R18W4TransferAgent):
 def setup(self,path):
  super().setup(path)
  # Close the original idle worker before assigning the new isolated executor.
  send_packet(self.worker.stdin,dict(close=True));self.worker.wait(timeout=60)
  self.worker=subprocess.Popen([str(B/'scripts/cw_bridge_python.sh'),str(B/'scripts/revision_v5_twentyeighth_native_executor.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.worker_log)
  p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(execution_model='Published carla_garage48e6a6d bicycle with official W4S7814310e PID/creep; actual controls unchanged;20Hz prediction sampled10Hz for EC',history='Own selected prediction under this executor; controller snapshot uses past errors and current own speed only',research_variant='NATIVE_EXECUTION_V01',dynamics_scope='0.5s Dev10 execution-prefix improvement established;4s predictions remain approximation',state_view='Math XY/yaw, actual rear offset; velocity and finite-difference world acceleration projected into body axes; initial observed state retained');p.write_text(json.dumps(m,indent=2)+'\n')
 def _replan(self,data,timestamp,pose):
  v=self.hero_actor.get_velocity();self.current_state['_native_controller']=snapshot(self,math.hypot(v.x,v.y));self.current_state['_native_controller']['plan_t0_seconds']=float(timestamp);self.current_state['_sensor_frame_ids']={k:int(data[k][0])for k in B2D_ORDER};return super()._replan(data,timestamp,pose)
