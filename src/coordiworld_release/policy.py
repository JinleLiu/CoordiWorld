"""Frozen PNC readout on an already generated candidate bank.

Candidate generation and native execution are supplied separately in the
research source; this interface does not infer EC from evaluator labels.
"""
from pathlib import Path
import numpy as np
from .tree import predict
from .metrics import moment_projection

HEADS=['q','qp','qttc','qlk','dac_event','physical_nc']

def evaluate(features,model_root,ec=None,history_supply=False,progress_override=True):
    """Return raw heads, projected moments, eligibility, scores, and selected index.

    `ec` must be the candidate-execution comfort measurement against a legal
    previous selected plan, in candidate order. Pass None for an initial frame.
    Supply is enabled only after the upstream history/candidate contract holds.
    """
    x=np.asarray(features['x674'],np.float32)
    valid=np.asarray(features['valid'],bool)
    geometry=int(features['geo'])
    if not valid.any() or not valid[geometry]:raise ValueError('Geometry reference must be valid.')
    raw={}
    for head in HEADS:
        extra={'q':'V24','dac_event':'R8','physical_nc':'C32'}.get(head)
        inputs=np.concatenate([x,features[extra]],axis=1) if extra else x
        raw[head]=predict(Path(model_root)/head,inputs)
    projected=moment_projection(np.column_stack([raw[k] for k in HEADS[:4]]))
    q,qp,qt,ql=projected.T
    local=(5*qp+5*qt+2*ql+2*np.asarray(features['features198'])[:,197]*q)/14
    dac=valid&(raw['dac_event']<=raw['dac_event'][geometry])
    eligible=dac&(raw['physical_nc']<=.5)
    if not eligible.any():eligible=dac&(raw['physical_nc']<=raw['physical_nc'][geometry])
    score=local.copy()
    if ec is not None:
        ec=np.asarray(ec,float)
        score=(14*local+2*q*ec)/16
        if history_supply:
            appended=np.arange(len(valid))>=12
            eligible|=appended&valid&(ec>0)&(raw['dac_event']<=.5)&(raw['physical_nc']<=.5)
            if progress_override:
                known=appended&(x[:,668]>0)
                score[known]+=5/16*(q[known]*x[known,662]-qp[known])
    if not eligible.any():raise ValueError('No eligible reference or candidate.')
    return {'choice':int(np.where(eligible,score,-np.inf).argmax()),'raw':raw,
            'projected':projected,'eligible':eligible,'scores':score,'local_scores':local}
