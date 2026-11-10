"""B16 proposed joint moments with known current/history comfort; NumPy only.

This defines targets, scores, fixed output feasibility maps, and row weights.
No training occurs, no uploaded code is imported, and no DEV labels are read.
Components are [NC,DAC,DDC,TLC,P,TTC,LK,HC], using existing human-filter labels.
The deployment HC comes from the verified current/history execution interface.

Q=NC*DAC*DDC*TLC; Z=Q*(5P+5TTC+2LK)/12.
Learn q=E[Q|X], z=E[Z|X], then S=(12*z+2*H*q)/14.
This follows linearity of expectation with observed H, not metric independence.
Targets are absolute moments. A matched scalar control must learn absolute U;
the frozen B14 regression instead predicts U-U_ref and is an additional control.
"""
from pathlib import Path
import json
import numpy as np


def targets(components):
    c = np.asarray(components, dtype=np.float64)
    if c.shape[-1] != 8 or np.any(~np.isfinite(c)):
        raise ValueError('expected finite [...,8] components')
    if np.any((c < 0) | (c > 1)):
        raise ValueError('components must retain their original [0,1] values')
    q = c[..., :4].prod(axis=-1)
    qp, qt, ql = q*c[...,4], q*c[...,5], q*c[...,6]
    z = (5*qp + 5*qt + 2*ql) / 12
    return {'q': q, 'z': z, 'qp': qp, 'qttc': qt, 'qlk': ql}


def score_two(q, z, current_hc):
    q,z,h=np.broadcast_arrays(np.asarray(q,float),np.asarray(z,float),np.asarray(current_hc,float))
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(z)) or np.any(~np.isfinite(h)):
        raise ValueError('nonfinite model output or HC')
    if np.any((h<0)|(h>1)):
        raise ValueError('HC must be in [0,1]')
    return (12*z+2*h*q)/14


def score_four(q, qp, qttc, qlk, current_hc):
    q,p,t,l,h=np.broadcast_arrays(*[np.asarray(x,float) for x in (q,qp,qttc,qlk,current_hc)])
    if any(np.any(~np.isfinite(x)) for x in (q,p,t,l,h)):
        raise ValueError('nonfinite model output or HC')
    if np.any((h<0)|(h>1)):
        raise ValueError('HC must be in [0,1]')
    return (5*p+5*t+2*l+2*h*q)/14


def project_qz(q_raw, z_raw):
    """Exact Euclidean projection onto 0 <= z <= q <= 1, in (q,z) units.

    First pool an order violation using the RAW mean, then clip both values.
    Clipping first and then averaging is a different, generally wrong map.
    E.g. raw q=-1,z=.5 projects to (0,0), not (.25,.25).
    A projection in (q,r=12z/14) coordinates is different. Register these units.
    Projection enforces feasible outputs; it does not guarantee better selection.
    """
    q,z=np.broadcast_arrays(np.asarray(q_raw,float),np.asarray(z_raw,float))
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(z)):
        raise ValueError('nonfinite raw moments')
    violate=z>q
    pooled=(q+z)/2
    return np.clip(np.where(violate,pooled,q),0,1),np.clip(np.where(violate,pooled,z),0,1)


def clip_four_feasible(q_raw, qp_raw, qttc_raw, qlk_raw):
    """Fixed feasible clipping, NOT a joint orthogonal projection.

    Enforce 0<=q<=1 and 0<=qp,qttc,qlk<=q. The true joint moments satisfy this.
    This operation neither assumes independent metrics nor hard-vetoes HC=0.
    """
    q,p,t,l=np.broadcast_arrays(*[np.asarray(x,float) for x in (q_raw,qp_raw,qttc_raw,qlk_raw)])
    if any(np.any(~np.isfinite(x)) for x in (q,p,t,l)):
        raise ValueError('nonfinite raw moments')
    q=np.clip(q,0,1)
    return q,np.clip(p,0,q),np.clip(t,0,q),np.clip(l,0,q)


def fit_row_weights(drives, valid):
    """Mean-one row weights for ONE fit's scene arrays, retaining valid order.

    Caller first subsets/arranges the fit scenes and applies the same row order
    to every feature/target array. All fit scenes count, including singletons
    and flat-label queries. Raw row weight is 1/(fit-drive-scenes * scene-M).
    This is the same drive/scene/candidate weighting as the B14 regressor.
    It is different from pairwise query weighting. Apply once in sample_weight.
    """
    d=np.asarray(drives);v=np.asarray(valid,bool)
    if v.ndim!=2 or d.shape!=(len(v),) or np.any(v.sum(1)==0):
        raise ValueError('each fit scene must contain valid candidates')
    _,inverse,counts=np.unique(d,return_inverse=True,return_counts=True)
    scene=1.0/(counts[inverse]*v.sum(1))
    rows=np.broadcast_to(scene[:,None],v.shape)[v].copy()
    return rows/rows.mean()


def synthetic_validation():
    rng=np.random.default_rng(170627)
    c=rng.random((1000,8));c[:,:4]=rng.choice([0.,.5,1.],(1000,4));c[:,5:]=rng.integers(0,2,(1000,3))
    t=targets(c);u=c[:,:4].prod(-1)*(c[:,4:]@np.array([5.,5.,2.,2.]))/14
    e2=float(abs(score_two(t['q'],t['z'],c[:,7])-u).max())
    e4=float(abs(score_four(t['q'],t['qp'],t['qttc'],t['qlk'],c[:,7])-u).max())
    p=project_qz(t['q'],t['z']);preserve=float(max(abs(p[0]-t['q']).max(),abs(p[1]-t['z']).max()))
    x=rng.uniform(-2,3,(1000,2));q,z=project_qz(x[:,0],x[:,1]);p=np.c_[q,z]
    vertices=np.array([[0.,0.],[1.,0.],[1.,1.]])
    vi=np.einsum('ij,ikj->ik',x-p,vertices[None,:,:]-p[:,None,:])
    vi_max=float(vi.max());feasible=bool(np.all((q>=0)&(q<=1)&(z>=0)&(z<=q)))
    reproj=project_qz(q,z);idem=float(max(abs(reproj[0]-q).max(),abs(reproj[1]-z).max()))
    w=rng.random(1000);w/=w.sum();h=0.
    expectation=float(abs(w@score_two(t['q'],t['z'],h)-score_two(w@t['q'],w@t['z'],h)))
    trade=np.ones((2,8));trade[0,5]=0;trade[1,7]=0
    tr=targets(trade);trade_scores=score_two(tr['q'],tr['z'],trade[:,7])
    ds=np.array(['a','a','b']);v=np.array([[1,1,0],[1,0,0],[1,1,1]],bool)
    rw=fit_row_weights(ds,v);row_drive=np.repeat(ds,v.sum(1));sums={a:float(rw[row_drive==a].sum()) for a in np.unique(ds)}
    assert e2<1e-14 and e4<1e-14 and preserve<1e-14 and feasible and vi_max<1e-12 and idem<1e-14
    assert expectation<1e-14 and trade_scores[1]>trade_scores[0] and abs(sums['a']-sums['b'])<1e-14
    return {'two_head_reconstruction_max_abs':e2,'four_head_reconstruction_max_abs':e4,
            'projection_preserves_feasible_truth_max_abs':preserve,'projection_feasible':feasible,
            'projection_variational_inequality_max':vi_max,'projection_idempotence_max_abs':idem,
            'conditional_expectation_identity_error':expectation,'beneficial_TTC_HC_trade_scores':trade_scores.tolist(),
            'fit_weight_total_by_drive':sums,'fit_row_weight_mean':float(rw.mean()),
            'scope':'Synthetic algebra/weights/projection only. No model fitting or performance prediction.'}


def train_label_validation(npz_path):
    # Explicit TRAIN keys only. Does not access DEV arrays/labels or fit a model.
    with np.load(npz_path,allow_pickle=False) as a:
        c=a['components'].astype(float);v=a['train_valid'];h=a['train_features'][...,197].astype(float);u=a['utility'].astype(float)
    t=targets(c);raw=c[...,:4].prod(-1)*(c[...,4:]*[5.,5.,2.,2.]).sum(-1)/14
    rec2=score_two(t['q'],t['z'],h);rec4=score_four(t['q'],t['qp'],t['qttc'],t['qlk'],h)
    hc_mismatch=int(((h!=c[...,7])&v).sum())
    return {'valid_candidates':int(v.sum()),'HC_input_label_mismatches':hc_mismatch,
            'two_head_max_abs_vs_formula':float(abs(rec2-raw)[v].max()),
            'four_head_max_abs_vs_formula':float(abs(rec4-raw)[v].max()),
            'two_head_max_abs_vs_stored_float32_U':float(abs(rec2-u)[v].max()),
            'scope':'All TRAIN candidate truth recomposition; no learned predictions, no DEV, no new performance result.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--train-npz');p.add_argument('--output');args=p.parse_args()
    result={'synthetic':synthetic_validation()}
    if args.train_npz:result['train_truth']=train_label_validation(args.train_npz)
    s=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:Path(args.output).write_text(s)
    print(s,end='')
