"""Causal static-component interface for the whole-method rebuild.

This module does not predict future signals/traffic, run native scoring, normalize
official progress, or apply the privileged human-future penalty filter.
It consumes full current-map membership from a separately verified exporter.
The legacy 24-token bounding summaries are explicitly inadmissible.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CurrentMapProvenance:
    scene_id: str
    observation_timestamp_us: int
    latest_source_timestamp_us: int
    geometry_sha256: str
    route_sha256: str
    complete_query_geometry: bool
    route_resolved: bool
    executed_envelope_covered: bool
    future_fields_accessed: bool = False

    def validate(self):
        if not self.scene_id or self.latest_source_timestamp_us > self.observation_timestamp_us:
            raise ValueError('Current/history source identity or timestamp invalid')
        for digest in [self.geometry_sha256,self.route_sha256]:
            if len(digest)!=64 or any(c not in 'study_scene_000027' for c in digest):
                raise ValueError('Serialized geometry and route identity required')
        if not (self.complete_query_geometry and self.route_resolved and self.executed_envelope_covered):
            raise ValueError('Incomplete current geometry/route/coverage: retain unsupported scene, do not infer safety')
        if self.future_fields_accessed:
            raise ValueError('Privileged future cannot enter deployment input')

def project_polyline(points, polyline):
    """Arc-length projection on the full current route centerline; earliest tie."""
    points=np.asarray(points,dtype=np.float64);line=np.asarray(polyline,dtype=np.float64)
    if points.shape[-1:]!=(2,) or line.ndim!=2 or line.shape[1]!=2 or len(line)<2:
        raise ValueError('Expected xy points and complete xy polyline')
    if not np.isfinite(points).all() or not np.isfinite(line).all():raise ValueError('Nonfinite route geometry')
    segments=np.diff(line,axis=0);length=np.linalg.norm(segments,axis=-1);keep=length>0
    if not keep.any():raise ValueError('Degenerate centerline')
    starts=line[:-1][keep];segments=segments[keep];length=length[keep]
    arc=np.r_[0.,np.cumsum(length)[:-1]]
    flat=points.reshape(-1,2);offset=flat[:,None]-starts[None]
    f=np.clip(np.einsum('nki,ki->nk',offset,segments)/length[None]**2,0,1)
    residual=offset-f[:,:,None]*segments[None];index=(residual**2).sum(-1).argmin(-1)
    return (arc[index]+f[np.arange(len(flat)),index]*length[index]).reshape(points.shape[:-1])

def evaluate_static(center_xy, corners_in_drivable_union, center_in_route_lanes,
                    center_in_intersection, route_centerline_xy, provenance):
    """Unfiltered drivable/direction and raw progress on 41 causal executed states.

    Membership must use the pinned native map layers and boundary predicate on
    full polygons. Those inputs cannot be replaced by centroid/extent tokens.
    Returned fields are features/intermediate components, not final label truth.
    """
    provenance.validate()
    center=np.asarray(center_xy,dtype=np.float64)
    corners=np.asarray(corners_in_drivable_union);onroute=np.asarray(center_in_route_lanes);inter=np.asarray(center_in_intersection)
    if center.ndim!=3 or center.shape[1:]!=(41,2) or not np.isfinite(center).all():raise ValueError('Require full 0..4s controller execution at .1s')
    if corners.shape!=center.shape[:2]+(4,) or onroute.shape!=center.shape[:2] or inter.shape!=onroute.shape:
        raise ValueError('Map memberships must match every executed state and corner')
    if any(x.dtype!=np.bool_ for x in [corners,onroute,inter]):raise ValueError('Explicit known boolean membership required')
    drivable=np.all(corners,axis=(1,2)).astype(float)
    distance=np.zeros(center.shape[:2]);distance[:,1:]=np.linalg.norm(np.diff(center,axis=1),axis=-1)
    distance[onroute|inter]=0
    # Preserve native inclusive [time_idx-10, time_idx] window exactly.
    cum=np.concatenate([np.zeros((len(center),1)),np.cumsum(distance,axis=1)],axis=1)
    right=np.arange(41)+1;left=np.maximum(0,np.arange(41)-10)
    max_oncoming=(cum[:,right]-cum[:,left]).max(-1)
    direction=np.where(max_oncoming<2.,1.,np.where(max_oncoming<6.,.5,0.))
    projected=project_polyline(center[:,[0,-1]],route_centerline_xy)
    raw_progress=np.maximum(0.,projected[:,1]-projected[:,0])
    return dict(unfiltered_drivable=drivable,unfiltered_driving_direction=direction,
                raw_centerline_progress_m=raw_progress,max_oncoming_window_m=max_oncoming,
                official_normalized_progress=None,official_filtered_components=None,
                traffic_light_future_compliance=None,
                semantics='Current-map intermediates; official normalized/filtered values require separate prediction and evaluation')
