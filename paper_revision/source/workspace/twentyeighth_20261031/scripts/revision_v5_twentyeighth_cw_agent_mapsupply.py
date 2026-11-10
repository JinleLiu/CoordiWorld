"""Matched v05 BaseEC arms: optional spatial supply, no H3/P0/relaxed gates."""
import hashlib,json
from pathlib import Path
from revision_v5_twentyeighth_cw_agent_lowspeed import CoordiWorldLowSpeedAgent
from revision_v5_twentyeighth_carla_view import pose
import revision_v5_twentyeighth_cw_agent as base
from revision_v5_twentyeighth_reference_runway import install as runway,CONFIG as RUNWAY
from revision_v5_twentyeighth_map_supply_policy import install as supply
from revision_v5_twentyeighth_map_supply import CONFIG

def get_entry_point():return 'CoordiWorldMapSupplyAgent'
class CoordiWorldMapSupplyAgent(CoordiWorldLowSpeedAgent):
    def setup(self,path):
        original=base.frame
        def frame(*args,**kwargs):
            h=original(*args,**kwargs);route=args[4];h['navigation_goal_math']=pose(route[-1][0]).tolist();return h
        base.frame=frame;super().setup(path)
        assert self.cfg['variant']=='BaseEC'
        if self.cfg.get('map_supply',False):supply(self.policy,self.hero_actor.get_world().get_map())
        runway(self.policy)
        p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(method=self.cfg.get('method_id',m['method']),adapter='v05_current_map_reference_runway',reference_runway=RUNWAY,map_supply=CONFIG if self.cfg.get('map_supply',False)else None,decision='Original BaseEC frozen six heads, risk qualification, EC score and argmax; P0/H3 disabled',navigation_input='Already provided route final waypoint; not future actor information');p.write_text(json.dumps(m,indent=2)+'\n')
