"""Current-only stopped-vehicle footprint guard; no learned or future inputs.

The 0.05 m/s stopped boundary is the existing pinned C32/official stopped
convention. Initial overlapping objects are excluded, explicitly recorded.
This is a conservative frozen-current hypothesis, not a traffic forecast.
"""
import numpy as np


def rectangle_gap(a, b):
    """Signed maximum SAT separating gap, metres; negative means overlap."""
    ca, sa = np.cos(a[..., 2]), np.sin(a[..., 2])
    cb, sb = np.cos(b[..., 2]), np.sin(b[..., 2])
    ua = np.stack([ca, sa], -1); va = np.stack([-sa, ca], -1)
    ub = np.stack([cb, sb], -1); vb = np.stack([-sb, cb], -1)
    d = b[..., :2] - a[..., :2]
    gaps = []
    for axis in [ua, va, ub, vb]:
        ra = np.abs((axis * ua).sum(-1)) * a[..., 3] / 2 + np.abs((axis * va).sum(-1)) * a[..., 4] / 2
        rb = np.abs((axis * ub).sum(-1)) * b[..., 3] / 2 + np.abs((axis * vb).sum(-1)) * b[..., 4] / 2
        gaps.append(np.abs((d * axis).sum(-1)) - ra - rb)
    return np.maximum.reduce(gaps)


def stopped_vehicle_guard(execution_local_pose, ego_speed, current, types,
                          current_known, valid, ego_box, rear_axle_offset):
    """M,T,3 local rear-axle poses and current N,7 centre actor states."""
    ex = np.asarray(execution_local_pose, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    n, m, t = len(cur), len(ex), ex.shape[1]
    support = np.asarray(current_known, bool) & (np.asarray(types) == 1)
    support &= np.linalg.norm(cur[:, 3:5], axis=-1) <= .05
    support &= np.isfinite(cur).all(-1) & (cur[:, 5] > 0) & (cur[:, 6] > 0)
    first = np.full(m, -1, dtype=int); actor = np.full(m, -1, dtype=int)
    if not n:
        return dict(unsafe=np.zeros(m, bool), first_step=first, first_actor=actor,
                    minimum_gap=np.full(m, np.nan), support=support,
                    initial_overlap=np.zeros((m, 0), bool))
    heading = np.stack([np.cos(ex[..., 2]), np.sin(ex[..., 2])], -1)
    center = ex[..., :2] + float(rear_axle_offset) * heading
    eb = np.concatenate([center, ex[..., 2:3], np.broadcast_to(ego_box, (m, t, 2))], -1)
    ab = np.concatenate([cur[:, :3], cur[:, 5:7]], -1)
    gap = rectangle_gap(eb[:, :, None, :], ab[None, None, :, :])
    initial = gap[:, 0] <= 0
    eligible = support[None, None, :] & ~initial[:, None, :]
    eligible = eligible & (np.asarray(ego_speed) > .05)[:, :, None]
    eligible[:, 0] = False
    contact = (gap <= 0) & eligible & np.asarray(valid, bool)[:, None, None]
    hit = contact.any((1, 2))
    for j in np.flatnonzero(hit):
        ti, ai = np.argwhere(contact[j])[0]; first[j] = int(ti); actor[j] = int(ai)
    minimum = np.where(eligible, gap, np.inf).min((1, 2))
    minimum[~np.isfinite(minimum)] = np.nan
    return dict(unsafe=hit, first_step=first, first_actor=actor, minimum_gap=minimum,
                support=support, initial_overlap=initial)
