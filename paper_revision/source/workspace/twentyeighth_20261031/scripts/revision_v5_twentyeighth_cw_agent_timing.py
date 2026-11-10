import json
from revision_v5_twentyeighth_cw_agent_mapsupply import CoordiWorldMapSupplyAgent
from revision_v5_twentyeighth_map_timing_policy import install,CONFIG
from revision_v5_twentyeighth_metric_capture import ActualMetricCapture
def get_entry_point():return 'CoordiWorldTimingMetrics'
class CoordiWorldTimingAgent(CoordiWorldMapSupplyAgent):
    def setup(self,path):
        super().setup(path);install(self.policy,self.hero_actor.get_world().get_map());p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(map_timing=CONFIG,decision='Original frozen risk qualification and score; MapSupply_v04 plus two slow8m entry timings only when its Parking spatial candidates are all rejected; no terminal guard, no P0');p.write_text(json.dumps(m,indent=2)+'\n')
class CoordiWorldTimingMetrics(ActualMetricCapture,CoordiWorldTimingAgent):pass
