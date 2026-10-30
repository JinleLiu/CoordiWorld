"""Portable candidate-value pilot. Inputs are frozen current-only world features.

No candidate label, future mask or simulator state is an inference argument.
The reference is the first valid candidate in the unchanged shared candidate pool.
Risk outputs denote filtered NC/DAC penalties, not universal physical collision.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

MODES = ['no_world', 'CTRA', 'factual', 'factual_plus_v8']
SEEDS = [17, 29, 43]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def world_features(features, mode):
    index = {'CTRA': 1, 'factual': 2, 'factual_plus_v8': 3, 'no_world': 2}[mode]
    x = features[:, index].copy()
    if mode == 'no_world':
        x[..., 64:120] = 0  # 8 times x 7 dynamic-world geometry features.
    return x


class CandidateValue(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.encoder = nn.Sequential(nn.Linear(188, 64), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(64, 32), nn.GELU())
        self.events_components = nn.Linear(32, 10)
        self.value = nn.Linear(32, 1) if family == 'direct_joint' else nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, reference):
        h = self.encoder(x)
        logits = self.events_components(h)
        index = torch.arange(len(x), device=x.device)
        if self.family == 'direct_joint':
            value = self.value(h).squeeze(-1).sigmoid()
            gain = value - value[index, reference, None]
        else:
            anchor = h[index, reference, None].expand_as(h)
            pair = torch.cat((h, anchor, h-anchor), -1)
            base = torch.cat((anchor, anchor, torch.zeros_like(anchor)), -1)
            gain = self.value(pair).squeeze(-1) - self.value(base).squeeze(-1)
            value = None
        return gain, logits, value


def scene_weights(drives):
    unique, inverse, counts = np.unique(drives, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def fit_model(features, valid, truth, drives, indices, family, seed, steps=400):
    torch.manual_seed(seed)
    sample = features[indices][valid[indices]]
    mean = sample.mean(0).astype(np.float32)
    scale = sample.std(0).clip(.1).astype(np.float32)
    x = torch.from_numpy(np.clip((features[indices]-mean)/scale, -8, 8)).cuda()
    v = torch.from_numpy(valid[indices]).cuda()
    ref = v.int().argmax(1)
    index = torch.arange(len(x), device=x.device)
    y = {k: torch.from_numpy(z[indices]).cuda() for k,z in truth.items()}
    delta = y['utility'] - y['utility'][index, ref, None]
    w = torch.from_numpy(scene_weights(drives[indices]).astype(np.float32)).cuda()
    vf = v.float(); den = vf.sum(1).clamp_min(1)
    def reduce(z): return ((z*vf).sum(1)/den*w).sum()
    model = CandidateValue(family).cuda()
    with torch.no_grad():
        prior = torch.cat((y['components'],y['events']),-1)
        p = ((prior*vf[...,None]).sum(1)/den[:,None]*w[:,None]).sum(0).clamp(.01,.99)
        model.events_components.bias.copy_(torch.logit(p))
        model.events_components.weight.zero_()
        last = model.value if family=='direct_joint' else model.value[-1]
        last.weight.zero_();last.bias.zero_()
        if family=='direct_joint':last.bias.fill_(torch.logit((y['utility']*vf*w[:,None]/den[:,None]).sum().clamp(.01,.99)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        gain, logits, value = model(x,ref)
        components = reduce(nn.functional.binary_cross_entropy_with_logits(logits[...,:8],y['components'],reduction='none').mean(-1))
        events = reduce(nn.functional.binary_cross_entropy_with_logits(logits[...,8:],y['events'],reduction='none').mean(-1))
        relative = reduce((gain-delta).square())
        preference = reduce(nn.functional.binary_cross_entropy_with_logits(gain/.1, torch.sigmoid(delta/.1),reduction='none'))
        absolute = reduce((value-y['utility']).square()) if value is not None else relative.new_zeros(())
        loss = components + events + 2*relative + 2*absolute + .1*preference
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step in [0,99,199,399,steps-1]:history.append(dict(update=step+1,loss=float(loss),components=float(components),events=float(events),relative=float(relative),absolute=float(absolute),preference=float(preference)))
    model.eval()
    return model,mean,scale,history


def predict(model, features, valid, mean, scale):
    with torch.inference_mode():
        x=torch.from_numpy(np.clip((features-mean)/scale,-8,8).astype(np.float32)).cuda()
        ref=torch.from_numpy(valid.argmax(1)).cuda()
        gain,logits,_=model(x,ref)
    gain=gain.cpu().numpy().astype(np.float64);logits=logits.cpu().numpy().astype(np.float64)
    return gain,logits


def probabilities(logits):
    return 1/(1+np.exp(-np.clip(logits,-50,50)))


def choose(gain, logits, valid, margin, risk_guard=True):
    refs=valid.argmax(1);ii=np.arange(len(refs));p=probabilities(logits)
    eligible=valid & (gain>margin)
    if risk_guard:
        eligible &= (p[...,8:] <= p[ii,refs,None,8:]+.01).all(-1)
        eligible &= p[...,4] >= p[ii,refs,None,4]-.01
    ranked=np.where(eligible,gain,-np.inf)
    selection=np.where(eligible.any(1),ranked.argmax(1),refs)
    return selection


def macro(values, drives):
    values=np.asarray(values,dtype=np.float64)
    return float(np.mean([values[drives==d].mean() for d in sorted(set(drives))]))


def bootstrap(diff, drives):
    values=np.asarray([np.asarray(diff,dtype=np.float64)[drives==d].mean() for d in sorted(set(drives))])
    rng=np.random.default_rng(102611)
    return np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975]).tolist()


def risk_summary(risk, truth, drives, constants):
    output=[]
    for j in range(2):
        p=risk[:,j];y=truth[:,j];constant=constants[j]
        b=macro((p-y)**2,drives);base=macro((constant-y)**2,drives)
        bins=[];ece=0.
        for k in range(10):
            mask=(p>=k/10)&((p<(k+1)/10) if k<9 else (p<=1))
            count=int(mask.sum())
            row=dict(bin_low=k/10,bin_high=(k+1)/10,n=count,events=int(y[mask].sum()))
            if count:
                row.update(mean_probability=float(p[mask].mean()),event_rate=float(y[mask].mean()))
                ece+=count/len(y)*abs(row['mean_probability']-row['event_rate'])
            bins.append(row)
        output.append(dict(event=['filtered_NC','filtered_DAC'][j],Brier_drive_macro=b,
                           Brier_scene_micro=float(np.mean((p-y)**2)),
                           constant_prevalence=constant,constant_Brier_drive_macro=base,
                           BSS_drive_macro=1-b/base if base>0 else None,
                           ECE_scene_micro=ece,bins=bins))
    return output


def reporting_calibration(logits, selected, events, drives):
    ii=np.arange(len(selected));z=logits[ii,selected,8:];y=events[ii,selected]
    w=scene_weights(drives);result=[]
    for j in range(2):
        constants=float((y[:,j]*w).sum());positive=int(y[:,j].sum());eligible=positive>=2 and len(y)-positive>=2
        temperature=1.;best=None
        if eligible:
            options=[]
            for t in np.geomspace(.25,8,41):
                l=z[:,j]/t;loss=float(((np.logaddexp(0,l)-y[:,j]*l)*w).sum());options.append((loss,float(t)))
            best,temperature=min(options)
        result.append(dict(temperature=temperature,constant=constants,positive=positive,n=len(y),eligible=eligible,NLL=best))
    return result


def evaluate(selected, gain, logits, truth, valid, drives, calibration=None):
    ii=np.arange(len(selected));refs=valid.argmax(1)
    y=truth['utility'];ev=truth['events'];cp=truth['components'];delta=y[ii,selected].astype(float)-y[ii,refs]
    events=ev[ii,selected];change=events-ev[ii,refs];switch=selected!=refs
    calibrated_logits=logits.copy()
    if calibration:
        for j,c in enumerate(calibration):calibrated_logits[...,8+j]/=c['temperature']
    risk=probabilities(calibrated_logits)[ii,selected,8:]
    out=dict(scenes=len(selected),drives=len(set(drives)),score_points=100*macro(y[ii,selected],drives),
        reference_score_points=100*macro(y[ii,refs],drives),gain_points=100*macro(delta,drives),
        exploratory_drive_CI_points=bootstrap(100*delta,drives),
        switches=int(switch.sum()),switch_fraction=float(switch.mean()),
        switched_mean_gain_points=float(100*delta[switch].mean()) if switch.any() else None,
        improved=int((delta>1e-6).sum()),worsened=int((delta< -1e-6).sum()),ties=int((abs(delta)<=1e-6).sum()),
        NC=int(events[:,0].sum()),DAC=int(events[:,1].sum()),
        reference_NC=int(ev[ii,refs,0].sum()),reference_DAC=int(ev[ii,refs,1].sum()),
        NC_new=int((change[:,0]>0).sum()),NC_rescued=int((change[:,0]<0).sum()),
        DAC_new=int((change[:,1]>0).sum()),DAC_rescued=int((change[:,1]<0).sum()),
        progress=macro(cp[ii,selected,4],drives),progress_delta=macro(cp[ii,selected,4]-cp[ii,refs,4],drives),
        oracle_regret_points=100*macro(np.where(valid,y,-np.inf).max(1)-y[ii,selected],drives),
        by_drive={d:dict(n=int((drives==d).sum()),gain_points=float((100*delta)[drives==d].mean())) for d in sorted(set(drives))},
        selected_candidates=selected.tolist(),selected_utility=y[ii,selected].tolist(),
        selected_events=events.tolist(),selected_progress=cp[ii,selected,4].tolist(),selected_risk=risk.tolist())
    if calibration:out['selected_risk_metrics']=risk_summary(risk,events,drives,[r['constant'] for r in calibration])
    return out
