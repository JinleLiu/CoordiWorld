"""Frozen TRAIN-exploration RouteSignalQ overlay; never fetches observations.

Inputs: existing same-population IC0 predictions, f198/valid; current-causal
route_signal46; X674 and V24. The caller must match the registered model/input
identity. No model adoption or unseen-drive claim is implied by this adapter.
"""
import numpy as np
from revision_v5_eighteenth_projection import project

def apply_local_q(base_prediction,features198,valid,route_signal46,new_q):
    mask=np.asarray(valid,bool)&np.asarray(route_signal46)[...,[26,32,34]].astype(bool).any(-1)
    raw={h:np.asarray(base_prediction['raw_'+h]).copy()for h in ['q','qp','qttc','qlk']}
    assert raw['q'].shape==mask.shape==np.asarray(new_q).shape
    raw['q']=np.where(mask,new_q,raw['q']);p=project(raw)
    s=(5*p[...,1]+5*p[...,2]+2*p[...,3]+2*np.asarray(features198)[...,197]*p[...,0])/14
    eligible=np.asarray(base_prediction['eligible'],bool);assert not(eligible&~valid).any();assert eligible.any(-1).all()
    return dict(choice=np.where(eligible,s,-np.inf).argmax(-1),scores=s,eligible=eligible,signal_application_mask=mask,**{'raw_'+h:a for h,a in raw.items()})

class RouteSignalQ:
    def __init__(self,model_path):
        import xgboost as xgb
        self.model=xgb.Booster();self.model.load_model(model_path)
        assert int(self.model.num_features())==744
    def predict(self,base_prediction,x674,v24,features198,valid,route_signal46):
        x=np.concatenate([x674,v24,route_signal46],-1).astype(np.float32)
        assert x.shape[-1]==744
        q=self.model.inplace_predict(x.reshape(-1,744)).reshape(valid.shape)
        return apply_local_q(base_prediction,features198,valid,route_signal46,q)
