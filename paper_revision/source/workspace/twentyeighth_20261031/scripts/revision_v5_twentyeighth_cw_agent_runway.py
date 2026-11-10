"""Shared truncated-reference repair for Full/BaseEC, no new supply or risk rule."""
import json
import revision_v5_twentyeighth_cw_agent as base
from revision_v5_twentyeighth_carla_view import pose
from revision_v5_twentyeighth_cw_agent_lowspeed import CoordiWorldLowSpeedAgent
from revision_v5_twentyeighth_reference_runway import install,CONFIG
from revision_v5_twentyeighth_metric_capture import ActualMetricCapture
def get_entry_point():return 'CoordiWorldRunwayMetrics'
class CoordiWorldRunwayAgent(CoordiWorldLowSpeedAgent):
    def setup(self,path):
        original=base.frame
        def frame(*args,**kwargs):
            h=original(*args,**kwargs);h['navigation_goal_math']=pose(args[4][-1][0]).tolist();return h
        base.frame=frame;super().setup(path);install(self.policy)
        p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(reference_runway=CONFIG,adapter='v05_common_reference_runway',method=self.cfg['method_id'],scope='Common engineering repair; Full retains its original H3/P0 selection, BaseEC retains original no-P0 selection. No spatial supply or new guard.');p.write_text(json.dumps(m,indent=2)+'\n')
class CoordiWorldRunwayMetrics(ActualMetricCapture,CoordiWorldRunwayAgent):pass
