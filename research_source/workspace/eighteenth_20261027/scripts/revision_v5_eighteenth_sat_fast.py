"""Allocation-reduced scalar-coordinate SAT, same operation order as S64 SAT."""
import numpy as np
def sat(a,b):
    ca,sa,cb,sb=np.cos(a[...,2]),np.sin(a[...,2]),np.cos(b[...,2]),np.sin(b[...,2]);dx=b[...,0]-a[...,0];dy=b[...,1]-a[...,1];values=[]
    for vx,vy in [(ca,sa),(-sa,ca),(cb,sb),(-sb,cb)]:
        values.append(abs(dx*vx+dy*vy)-abs(ca*vx+sa*vy)*a[...,3]/2-abs(-sa*vx+ca*vy)*a[...,4]/2-abs(cb*vx+sb*vy)*b[...,3]/2-abs(-sb*vx+cb*vy)*b[...,4]/2)
    return np.maximum.reduce(values)
