"""Shared current-map height support; optional supply; no-new-stop witness."""
import json
import revision_v5_twentyeighth_cw_agent as base
from revision_v5_twentyeighth_cw_agent_mapsupply import CoordiWorldMapSupplyAgent
from revision_v5_twentyeighth_map_timing_policy import install as timing
from revision_v5_twentyeighth_continuation_guard import install as guard,CONFIG as GUARD
from revision_v5_twentyeighth_observed_support import filtered_frame,CONFIG as SUPPORT
from revision_v5_twentyeighth_metric_capture import ActualMetricCapture
def get_entry_point():return 'CoordiWorldSupportMetrics'
class CoordiWorldSupportAgent(CoordiWorldMapSupplyAgent):
    def setup(self,path):
        original=base.frame
        def frame(*args,**kwargs):return filtered_frame(original(*args,**kwargs),args[0].get_world().get_map())
        base.frame=frame;super().setup(path)
        if self.cfg.get('map_timing'):timing(self.policy,self.hero_actor.get_world().get_map())
        if self.cfg.get('continuation_guard'):guard(self.policy,self.hero_actor.get_world().get_map())
        p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(current_support=SUPPORT,continuation_guard=GUARD if self.cfg.get('continuation_guard')else None,decision='Original NC/DAC gates and score; shared current support and optional static continuation witness, no new stop, no P0');p.write_text(json.dumps(m,indent=2)+'\n')
class CoordiWorldSupportMetrics(ActualMetricCapture,CoordiWorldSupportAgent):pass
