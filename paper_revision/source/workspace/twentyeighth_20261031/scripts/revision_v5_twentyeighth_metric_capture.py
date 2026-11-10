"""Post-decision ego telemetry only; frozen policies never consume these records."""
import json,time
from pathlib import Path
class ActualMetricCapture:
 def setup(self,path):
  super().setup(path);cfg=json.load(open(path.split('+')[0]));self._mc_out=Path(cfg['output_dir']);self._mc_file=open(self._mc_out/'actual_metric_10hz.jsonl','w',buffering=1);self._mc_tick=0;self._mc_errors=[];self._mc_calls=0;self._mc_seconds=0.
  self._mc_spec=dict(protocol='raw pinned Bench2Drive AutonomousAgent.get_metric_info, after native control computed; every second20Hz callback',sample_interval_seconds=.1,phase='first callback then every2; includes native startup',permissions='current ego only, logging after decision; values never enter model/control',units=dict(acceleration='m/s^2 world',angular_velocity='CARLA degrees/s world',forward_vector='unit world',right_vector='unit world',location='m world',rotation='CARLA degrees roll/pitch/yaw'),analysis='raw official-script literal and any SI diagnostics separately labelled; older missing telemetry is not imputed')
  (self._mc_out/'metric_capture.json').write_text(json.dumps(self._mc_spec,indent=2)+'\n')
 def run_step(self,*args,**kwargs):
  control=super().run_step(*args,**kwargs)
  if self._mc_tick%2==0:
   start=time.perf_counter()
   try:
    row=self.get_metric_info();row['sim_seconds']=float(args[1]if len(args)>1 else kwargs['timestamp']);row['callback_index']=self._mc_tick;self._mc_file.write(json.dumps(row)+'\n');self._mc_calls+=1
   except Exception as e:self._mc_errors.append(dict(callback=self._mc_tick,error=repr(e)))
   self._mc_seconds+=time.perf_counter()-start
  self._mc_tick+=1;return control
 def destroy(self):
  if hasattr(self,'_mc_file'):
   self._mc_file.close();(self._mc_out/'metric_capture.json').write_text(json.dumps(self._mc_spec|dict(samples=self._mc_calls,errors=self._mc_errors,capture_seconds=self._mc_seconds),indent=2)+'\n')
  super().destroy()
