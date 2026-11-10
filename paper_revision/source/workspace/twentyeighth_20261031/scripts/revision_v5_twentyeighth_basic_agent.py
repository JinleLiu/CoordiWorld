"""R18 frozen25km/h BasicAgent; same CARLA implementation as R17, new cohort."""
import json
from revision_v5_twentyseventh_bench_smoke_agent import InfrastructureSmokeAgent
def get_entry_point():return 'R18BasicAgent'
class R18BasicAgent(InfrastructureSmokeAgent):
 def setup(self,path_to_conf_file):
  super().setup(path_to_conf_file)
  p=self.out/'agent_metadata.json';r=json.loads(p.read_text());r.update(role='R18_COMMON_COHORT_OFFICIAL_PLANNER',identity='official-planner-adapted-interface',method='BasicAgent25',eval_seed=self.cfg['eval_seed'],method_config_sha=self.cfg['method_config_sha']);p.write_text(json.dumps(r,indent=2)+'\n')
