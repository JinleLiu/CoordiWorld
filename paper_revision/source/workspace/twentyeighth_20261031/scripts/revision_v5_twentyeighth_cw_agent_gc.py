"""Decision-equivalent v02 bridge: isolate long-lived objects from repeated full GC."""
import gc,json,time,resource
from revision_v5_twentyeighth_cw_agent_fast import CoordiWorldFastAgent

def get_entry_point():return 'CoordiWorldGCAgent'
class CoordiWorldGCAgent(CoordiWorldFastAgent):
 def setup(self,path):
  super().setup(path);self._static_gc_done=False
 def run_step(self,data,timestamp):
  control=super().run_step(data,timestamp)
  if not self._static_gc_done and self.policy.n>=1:
   t=time.perf_counter();collected=gc.collect();gc.freeze();self._static_gc_done=True
   row=dict(mode='one-time permanent generation after first completed plan',seconds=time.perf_counter()-t,collected=collected,frozen_objects=gc.get_freeze_count(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,policy_replans=self.policy.n,sim_seconds=float(timestamp),scope='GC bookkeeping only; native gc.collect calls retained; one agent process per episode; numeric algorithms and controls unchanged',timing='one-time cost outside base-agent run_step trace, included in episode wall time')
   (self.out/'static_gc.json').write_text(json.dumps(row,indent=2)+'\n')
   p=self.out/'agent_metadata.json';m=json.load(open(p));m['static_gc']=row;p.write_text(json.dumps(m,indent=2)+'\n')
  return control
