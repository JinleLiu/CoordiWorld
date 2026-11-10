"""Post-selection observations only; no changes to input, gate or controls."""
import json,hashlib,time
import numpy as np
from pathlib import Path
from cw_controller_prefix import physics_receipt,snapshot,roll
class V05Observation:
    def setup(self,path):
        super().setup(path)
        self.v05_physics=physics_receipt(self.hero_actor,self.geometry)
        (self.out/'v05_physics.json').write_text(json.dumps(self.v05_physics,indent=2)+'\n')
        original=self.policy.plan
        def observed_plan(history):
            state=snapshot(self);n=self.policy.n
            ca,native,receipt=original(history)
            t=time.perf_counter();prefix,detail=roll(history[-1],ca,native,state,self.v05_physics)
            np.savez_compressed(self.out/'policy'/f'v05_selected_prefix_{n:05d}.npz',prefix=prefix,commands=detail['commands'],selected_candidate=ca,selected_native=native)
            record=dict(replan=n,choice=receipt['choice'],timestamp_us=receipt['timestamp_us'],source_sha=receipt['source_sha'],candidate_sha=hashlib.sha256(ca.tobytes()).hexdigest(),native_sha=hashlib.sha256(native.tobytes()).hexdigest(),controller_state_before_selection=state,prefix_model=detail,logging_seconds=time.perf_counter()-t,role='POST_SELECTION_DIAGNOSTIC_ONLY_NO_ELIGIBILITY_OR_CONTROL_EFFECT')
            with(self.out/'policy/v05_prefix.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
            return ca,native,receipt
        self.policy.plan=observed_plan
