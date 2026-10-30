"""Pure NumPy math for a B17 progress-head experiment; no model fitting.

Inputs must be causal current-state/candidate quantities. P0 is a deterministic
proxy, not official progress: raw Q, human filtering and reactive outcomes remain
unknown. Targets use the unchanged final human-filter component labels.
"""
import numpy as np

def physical_progress_proxy(candidate_distance_m, pdm_distance_m, known):
    """Pair-PDM proxy assuming both raw Q=1, with the pinned 5 m branch.

    Distances must use the same reconstructed current PDM centerline and actual
    executed candidate states. Unknown proxy is zero plus an explicit mask.
    This is not the old maximum-within-12-candidates proxy.
    """
    d,b,k=np.broadcast_arrays(np.asarray(candidate_distance_m,float),np.asarray(pdm_distance_m,float),np.asarray(known,bool))
    if np.any(~np.isfinite(d[k])) or np.any(~np.isfinite(b[k])):
        raise ValueError('known distances must be finite')
    d=np.maximum(np.where(k,d,0),0);b=np.maximum(np.where(k,b,0),0)
    den=np.maximum(d,b);out=np.ones_like(d)
    np.divide(d,den,out=out,where=den>5.)
    return np.where(k,np.clip(out,0,1),0),k

def residual_targets(final_components, p0):
    c=np.asarray(final_components,float);p0=np.asarray(p0,float)
    if c.shape[-1]!=8 or p0.shape!=c.shape[:-1]:raise ValueError('shape mismatch')
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(p0)):raise ValueError('nonfinite target')
    q=np.prod(c[...,:4],axis=-1)
    return q, q*(c[...,4]-p0)

def reconstruct_qp(q_hat,residual_hat,p0,clip=True):
    q,r,p=np.broadcast_arrays(np.asarray(q_hat,float),np.asarray(residual_hat,float),np.asarray(p0,float))
    q=np.clip(q,0,1);qp=p*q+r
    return np.clip(qp,0,q) if clip else qp

def centered_objective_scale(groups,row_weights,lam=1.):
    """One deterministic scale for ONE fit; depends on layout/weights only.

    Makes the mean diagonal upper Hessian equal to the lambda=0 square-error
    convention. This controls global loss scale, not full regularizer equivalence.
    """
    g,a=np.broadcast_arrays(np.asarray(groups),np.asarray(row_weights,float))
    if g.ndim!=1 or len(g)==0 or not np.isfinite(lam) or lam<0 or np.any(~np.isfinite(a)) or np.any(a<=0):
        raise ValueError('invalid fit layout or weights')
    _,inverse,counts=np.unique(g,return_inverse=True,return_counts=True)
    unscaled=a*(1+lam*(counts[inverse]>1))
    return float(a.sum()/unscaled.sum())

def centered_loss_gradient_upper_hessian(pred,target,groups,row_weights,lam=1.):
    """Absolute MSE + all-candidate difference MSE, using scene groups.

    L=c*.5*sum_j a_s * [e_j^2 + lam*(e_j-mean_s(e))^2],
    with constant a_s inside each scene. All-pairs squared difference equals
    2*M*sum centered_error^2. Returned hessian is a positive diagonal upper
    bound for XGBoost's custom-objective interface, not the full Hessian.
    c=sum(a)/sum(unscaled_upper) is fixed inside a fit, independent of errors.
    Loss, gradients, and Hessian bounds all get the same c. This preserves the
    mean-Hessian scale of lambda=0 without claiming equivalent regularization.
    Apply weights here once; do not also give weights to DMatrix. This math does
    not assert official XGBoost support/convergence for arbitrary query-coupled
    losses; the diagonal bound is an explicit optimization approximation.
    """
    p,t,g,a=np.broadcast_arrays(np.asarray(pred,float),np.asarray(target,float),np.asarray(groups),np.asarray(row_weights,float))
    if p.ndim!=1 or len(p)==0 or np.any(~np.isfinite(p)) or np.any(~np.isfinite(t)):
        raise ValueError('invalid objective arguments')
    scale=centered_objective_scale(g,a,lam)
    e=p-t;grad=np.empty_like(e);upper=np.empty_like(e);loss=0.
    for group in np.unique(g):
        ix=np.flatnonzero(g==group);ee=e[ix];ww=a[ix]
        if not np.allclose(ww,ww[0],rtol=0,atol=1e-12):raise ValueError('row weights must be constant inside a scene')
        centered=ee-ee.mean();loss+=.5*np.sum(ww*(ee**2+lam*centered**2))
        grad[ix]=ww*(ee+lam*centered)
        upper[ix]=ww*(1+lam if len(ix)>1 else 1)
    return float(scale*loss),scale*grad,scale*upper

def validate():
    rng=np.random.default_rng(20261027)
    c=rng.random((200,8));c[:,:4]=rng.choice([0,.5,1],size=(200,4))
    p0=rng.random(200);q,r=residual_targets(c,p0)
    exact=float(np.max(np.abs(reconstruct_qp(q,r,p0)-q*c[:,4])))
    # P0 is constant within each conditional distribution, so no independence.
    qq=rng.choice([0,.5,1],size=50);pp=rng.random(50);w=rng.random(50);w/=w.sum();baseline=.63
    conditional_error=float(abs(w@(qq*pp)-(baseline*(w@qq)+w@(qq*(pp-baseline)))))
    d=np.array([0.,2.,6.,8.,3.]);b=np.array([0.,4.,10.,6.,5.]);known=np.array([1,1,1,1,0],bool)
    p,k=physical_progress_proxy(d,b,known);np.testing.assert_allclose(p,[1,1,.6,1,0],rtol=0,atol=1e-14)
    groups=np.repeat(np.arange(5),[1,2,3,6,12]);n=len(groups);weights=np.repeat(rng.uniform(.2,2,5),[1,2,3,6,12]);pred=rng.normal(size=n);target=rng.normal(size=n)
    loss,grad,upper=centered_loss_gradient_upper_hessian(pred,target,groups,weights)
    scale=centered_objective_scale(groups,weights)
    zero_loss,zero_grad,zero_hess=centered_loss_gradient_upper_hessian(pred,target,groups,weights,lam=0)
    zero_graderr=float(abs(zero_grad-weights*(pred-target)).max())
    zero_hesserr=float(abs(zero_hess-weights).max())
    mean_hess_error=float(abs(upper.mean()-zero_hess.mean()))
    numerical=[];eps=1e-5
    for i in range(n):
        a=pred.copy();b=pred.copy();a[i]+=eps;b[i]-=eps
        numerical.append((centered_loss_gradient_upper_hessian(a,target,groups,weights)[0]-centered_loss_gradient_upper_hessian(b,target,groups,weights)[0])/(2*eps))
    graderr=float(np.max(abs(grad-numerical)));mineig=1.
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group);m=len(ix);H=scale*weights[ix[0]]*(2*np.eye(m)-np.ones((m,m))/m);D=np.diag(upper[ix]);mineig=min(mineig,float(np.linalg.eigvalsh(D-H).min()))
    singleton=np.flatnonzero(groups==0)
    singleton_graderr=float(abs(grad[singleton]-scale*weights[singleton]*(pred-target)[singleton]).max())
    singleton_hesserr=float(abs(upper[singleton]-scale*weights[singleton]).max())
    assert exact<1e-14 and conditional_error<1e-14 and graderr<1e-8 and mineig>-1e-12
    assert zero_graderr==0 and zero_hesserr==0 and centered_objective_scale(groups,weights,0)==1
    assert mean_hess_error<1e-14 and singleton_graderr<1e-14 and singleton_hesserr<1e-14
    return {'status':'PASS','scope':'Synthetic identities, gradients and Hessian upper bound only; no learned performance','exact_QP_reconstruction_error':exact,'conditional_expectation_error':conditional_error,'centered_loss_gradient_error':graderr,'minimum_eigenvalue_of_diagonal_upper_minus_true_Hessian':mineig,'fit_scale_at_lambda_1':scale,'lambda_zero_gradient_error':zero_graderr,'lambda_zero_Hessian_error':zero_hesserr,'mean_upper_Hessian_at_lambda_0':float(zero_hess.mean()),'mean_upper_Hessian_at_lambda_1':float(upper.mean()),'mean_Hessian_scale_error':mean_hess_error,'singleton_gradient_error':singleton_graderr,'singleton_Hessian_error':singleton_hesserr,'low_progress_branch_proxy':p.tolist()}

if __name__=='__main__':
    import json
    print(json.dumps(validate(),indent=2))
