"""One-sided current-route progress penalty for a frozen relative-value head.

Unknown map pairs receive zero penalty and retain the original value; unknown
never means safe. This proxy is not native future progress or a safety guarantee.
"""
import numpy as np

def select_with_progress_penalty(gain, valid, geometry, weight):
    if gain.shape != valid.shape or geometry.shape != (*valid.shape,18):
        raise ValueError('Expected aligned values, mask and18 current-map features')
    if not valid.any(1).all() or not np.isfinite(geometry).all() or not np.isfinite(weight) or weight<0:
        raise ValueError('Invalid progress penalty inputs')
    reference=valid.argmax(1);ii=np.arange(len(valid));known=geometry[...,0]>.5
    supported=known & known[ii,reference,None]
    deficit=np.maximum(geometry[ii,reference,None,4]-geometry[...,4],0)*supported
    adjusted=gain-weight*deficit
    choice=np.where(valid,adjusted,-np.inf).argmax(1)
    return choice,adjusted
