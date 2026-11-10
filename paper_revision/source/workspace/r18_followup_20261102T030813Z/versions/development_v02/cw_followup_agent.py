"""Original CARLA adapter/metrics, new common supply and independent readouts."""
import json,hashlib
from pathlib import Path
import revision_v5_twentyeighth_cw_agent as base
from revision_v5_twentyeighth_cw_agent_tight import CoordiWorldSupportAgent
from revision_v5_twentyeighth_metric_capture import ActualMetricCapture
from cw_controller_prefix import physics_receipt,snapshot,CONFIG as PREFIX_CONFIG
from cw_common_policy import install,independent_readout_initialization

def get_entry_point():return 'CoordiWorldFollowupMetrics'

class CoordiWorldFollowupAgent(CoordiWorldSupportAgent):
    def setup(self,path):
        cfg=json.load(open(path.split('+')[0]));assert not any(cfg.get(k,False) for k in ['map_supply','map_timing','continuation_guard'])
        with independent_readout_initialization(cfg['readout']=='Simple'):super().setup(path)
        self.supply_physics=physics_receipt(self.hero_actor,self.geometry)
        original=base.frame
        def observed_frame(*args,**kwargs):
            f=original(*args,**kwargs);f['controller_state']=snapshot(self);return f
        base.frame=observed_frame;install(self.policy,self)
        p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(method=self.cfg['method_id'],adapter='R18_followup_Supply_v2_L1_dev_v01',interface='L1 offline contract validated; actual episode prefixes gated',supply_mode=self.cfg['supply_mode'],readout=self.cfg['readout'],supply_physics=self.supply_physics,controller_prefix=PREFIX_CONFIG,actual_controller_unchanged=True,upstream_learning='frozen world model and candidate feature construction retained; CW heads used only by CW branch',decision='common observed geometry/physical prefix/static continuation before CW qualification or independent Simple rank',candidate_count='original12 plus at most1 shared physically checked deceleration/stop plus at most5 attempted supply proposals',source_shas={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('cw_*.py')})
        p.write_text(json.dumps(m,indent=2)+'\n')
        p=self.out/'policy/model_identity.json';m=json.load(open(p));m.update(readout=self.cfg['readout'],CW_head_checkpoint_weights_loaded=self.cfg['readout']=='CW',CW_head_predictions_used=self.cfg['readout']=='CW',upstream_frozen_world_retained=True);p.write_text(json.dumps(m,indent=2)+'\n')

class CoordiWorldFollowupMetrics(ActualMetricCapture,CoordiWorldFollowupAgent):pass
