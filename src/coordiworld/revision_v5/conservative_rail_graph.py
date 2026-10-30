"""Conservative broad phase for the existing flat-strip contact implementation.

No distance/nearest-K heuristic. Every eligible leader must intersect a strip
of the follower's complete path. Disjoint envelopes therefore imply that the
pair can never qualify under the frozen contact test, for any candidate/time.
Full path, body radii, initial body boxes and a 1mm numerical guard are retained.
"""
import numpy as np


def build_graph(paths, info, initial, capacity=None):
    n = len(info)
    capacity = capacity or max(8, (n + 7)//8*8)
    if n:
        xy = np.asarray(paths[:, :, :2], np.float64)
        lo, hi = xy.min(1), xy.max(1)
        # Contact strips extend by follower half width in an arbitrary heading.
        own_lo = lo-info[:, 4:5]/2-.001
        own_hi = hi+info[:, 4:5]/2+.001
        # Swept strips lie between points on the full path, plus half width.
        # A body at any path position/heading fits inside this larger L1 radius.
        radius = (info[:, 3:4]+info[:, 4:5])/2+.001
        lead_lo = np.minimum(lo, initial[:, :2])-radius
        lead_hi = np.maximum(hi, initial[:, :2])+radius
        edge = ((own_hi[:, None] >= lead_lo[None]).all(-1) &
                (lead_hi[None] >= own_lo[:, None]).all(-1))
        # The legacy strip implementation assigns a zero tangent to degenerate
        # strips. That numerical behavior can report contacts anywhere. Preserve
        # its semantics by keeping every leader for such a follower.
        strip_ids = np.r_[np.arange(0,256,8),255]
        degenerate = (np.linalg.norm(np.diff(xy[:,strip_ids],axis=1),axis=-1)<1e-6).any(1)
        nonfinite = ~np.isfinite(xy).all((1,2)) | ~np.isfinite(info).all(1) | ~np.isfinite(initial).all(1)
        edge[degenerate|nonfinite] = True
        edge[:,nonfinite] = True
        np.fill_diagonal(edge, False)
        degree = edge.sum(1)
        k = max(1, int(degree.max()))
    else:
        edge = np.zeros((0, 0), bool);degree=np.zeros(0,int);k=1
    # Small power-of-two buckets reduce compile variants without deleting edges.
    bucket = min(capacity, max(8, 2**int(np.ceil(np.log2(k)))))
    indices = np.zeros((capacity, bucket), np.int64)
    valid = np.zeros((capacity, bucket), bool)
    for i in range(n):
        selected = np.flatnonzero(edge[i])
        indices[i, :len(selected)] = selected
        valid[i, :len(selected)] = True
    return indices, valid, {'actors':n, 'capacity':capacity, 'neighbors':bucket,
                           'edges':int(edge.sum()), 'possible_directed_edges':n*max(n-1,0),
                           'max_degree':int(degree.max()) if n else 0,
                           'all_actors_retained':True, 'no_nearest_K_cutoff':True}
