"""41-step current geometry traces from audited bundles and causal execution.

No raw-log reader, MetricCache, future traffic or target interface is imported.
All query polygons are consulted. Unsupported geometry returns an explicit mask;
it does not remove a candidate or set its predicted risk to zero/one.
"""
import numpy as np
from current_map_export import canonical_hash,footprint

TRACE_FIELDS=('corner0','corner1','corner2','corner3','route_center',
              'intersection_center','current_red_center','current_green_center',
              'current_signal_observed_center','raw_arc_delta_div100',
              'centerline_distance_div100','static_known')

def build_trace(bundle,executed_local):
    import shapely
    from shapely.geometry import LineString
    c=bundle['current'];p=bundle['provenance']
    if c['latest_source_timestamp_us']>c['timestamp_us'] or p['future_fields_accessed']:
        raise ValueError('Future source not deployable')
    if canonical_hash(bundle['polygons'])!=p['geometry_sha256'] or canonical_hash(bundle['route'])!=p['route_sha256']:
        raise ValueError('Full map source hash mismatch')
    e=np.asarray(executed_local,dtype=float)
    if e.ndim!=3 or e.shape[1:]!=(41,3) or not np.isfinite(e).all():raise ValueError('Causal 41-step poses required')
    actions=np.concatenate([e[...,:2]/100,np.cos(e[...,2:3]),np.sin(e[...,2:3]),np.broadcast_to(np.linspace(0,1,41)[None,:,None],(len(e),41,1))],-1).astype('float32')
    trace=np.zeros((len(e),41,12),dtype='float32')
    supported=p['complete_query_geometry'] and p['route_resolved'] and p['executed_envelope_covered']
    if not supported:return actions,trace
    origin=np.asarray(c['rear_axle_pose']);co,si=np.cos(origin[2]),np.sin(origin[2]);g=e.copy();g[...,:2]=e[...,:2]@np.array([[co,si],[-si,co]])+origin[:2];g[...,2]+=origin[2]
    v=bundle['vehicle'];coords=footprint(g,v['length'],v['width'],v['rear_axle_to_center'])
    route=set(bundle['route']['lane_ids']);lights={str(token):bool(red) for token,red in c['traffic_lights']}
    flags=np.zeros((len(e),41,9),bool)
    for record in bundle['polygons']:
        poly=shapely.from_wkb(bytes.fromhex(record['wkb_hex']))
        inside=shapely.contains_xy(poly,coords[...,0],coords[...,1]);layer=record['layer'];token=record['token']
        if layer in {'ROADBLOCK','INTERSECTION','DRIVABLE_AREA','CARPARK_AREA'}:flags[...,:4]|=inside[...,:4]
        if layer in {'LANE','LANE_CONNECTOR'} and token in route:flags[...,4]|=inside[...,4]
        if layer=='INTERSECTION':flags[...,5]|=inside[...,4]
        if layer=='LANE_CONNECTOR' and token in lights:
            flags[...,6 if lights[token] else 7]|=inside[...,4];flags[...,8]|=inside[...,4]
    center=shapely.points(coords[...,4,:]);line=LineString(bundle['route']['centerline_xy'])
    arc=shapely.line_locate_point(line,center)
    trace[...,:9]=flags;trace[...,9]=(arc-arc[:,0:1])/100.;trace[...,10]=shapely.distance(center,line)/100.;trace[...,11]=1.
    if not np.isfinite(trace).all():raise ValueError('Nonfinite current geometry')
    return actions,trace

def fold_normalization(records,meta,fit_ids):
    """Equal TRAIN scenes only. Caller must supply the exact OOF fit identities."""
    if len(records)!=len(meta):raise ValueError('Record identity mismatch')
    ids=set(fit_ids);byid={m['scene_id']:m for m in meta}
    if len(ids)!=len(fit_ids) or not ids or not ids<=set(byid):raise ValueError('Invalid fit identities')
    if any(byid[s]['split']!='train' for s in ids):raise ValueError('CAL or held-out rows cannot fit normalization')
    sums={k:np.zeros(n) for k,n in [('C',170),('A',44)]};squares={k:np.zeros_like(v) for k,v in sums.items()};count={'C':0,'A':0}
    for r,m in zip(records,meta):
        if m['scene_id'] not in ids:continue
        for k in sums:
            x=np.asarray(r[k],dtype=float) if k=='C' else np.asarray(r[k],dtype=float)[r['actor_mask']]
            if len(x):sums[k]+=x.mean(0);squares[k]+=(x*x).mean(0);count[k]+=1
    return {k:(sums[k]/max(count[k],1),np.maximum(np.sqrt(np.maximum(0,squares[k]/max(count[k],1)-(sums[k]/max(count[k],1))**2)),.05)) for k in sums}
