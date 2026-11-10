"""Separate shared terminal-viability guard, optionally with MapSupply."""
import json
from revision_v5_twentyeighth_cw_agent_mapsupply import CoordiWorldMapSupplyAgent
from revision_v5_twentyeighth_terminal_viability import install,CONFIG
def get_entry_point():return 'CoordiWorldViabilityAgent'
class CoordiWorldViabilityAgent(CoordiWorldMapSupplyAgent):
    def setup(self,path):
        super().setup(path);install(self.policy,self.hero_actor.get_world().get_map())
        p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(terminal_viability_guard=CONFIG,decision='Frozen BaseEC NC/DAC and scores, shared jerk-bounded stop candidate; separately reject eligible parking candidates with no current-static terminal continuation witness; if none, retain original eligible set',continuation_inputs='Predicted ego terminal11-state and current stationary boxes; current map; no observed future traffic',limitations='Static terminal witness is not a guarantee of future traffic clearance');p.write_text(json.dumps(m,indent=2)+'\n')
