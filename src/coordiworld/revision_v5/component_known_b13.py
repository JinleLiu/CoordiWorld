"""Separate geometry/envelope/route/progress knowledge; current inputs only."""
import numpy as np

FIELDS=('map_geometry_available','execution_envelope_available','route_resolved',
        'DAC_known','DAC_value','DDC_known','DDC_value',
        'progress_known','progress_proxy','history_comfort')

def from_legacy(geometry,comfort):
    known=geometry[...,0]>.5
    x=np.zeros((*known.shape,10),np.float32)
    for j in [0,1,2,3,5,7]:x[...,j]=known
    x[...,4]=np.where(known,geometry[...,1],0)
    x[...,6]=np.where(known,geometry[...,2],0)
    x[...,8]=np.where(known,geometry[...,4],0)
    x[...,9]=comfort
    return x

def partial_current_DAC(bundle,executed_local):
    """Do not let unknown route erase observed drivable-area geometry."""
    import shapely
    from current_map_export import canonical_hash,footprint
    c=bundle['current'];p=bundle['provenance'];e=np.asarray(executed_local,dtype=float)
    if c['latest_source_timestamp_us']>c['timestamp_us'] or p['future_fields_accessed']:raise ValueError('Future information forbidden')
    if canonical_hash(bundle['polygons'])!=p['geometry_sha256'] or canonical_hash(bundle['route'])!=p['route_sha256']:raise ValueError('Current map source hash mismatch')
    assert e.shape[1:]==(41,3) and np.isfinite(e).all()
    flags=dict(map_geometry_available=bool(p['complete_query_geometry']),execution_envelope_available=bool(p['executed_envelope_covered']),route_resolved=bool(p['route_resolved']))
    known=flags['map_geometry_available'] and flags['execution_envelope_available']
    if not known:return np.zeros(len(e)),flags
    origin=np.array(c['rear_axle_pose']);co,si=np.cos(origin[2]),np.sin(origin[2]);g=e.copy();g[...,:2]=e[...,:2]@np.array([[co,si],[-si,co]])+origin[:2];g[...,2]+=origin[2]
    v=bundle['vehicle'];corners=footprint(g,v['length'],v['width'],v['rear_axle_to_center'])[...,:4,:]
    inside=np.zeros(corners.shape[:-1],bool)
    for record in bundle['polygons']:
        if record['layer'] not in {'ROADBLOCK','INTERSECTION','DRIVABLE_AREA','CARPARK_AREA'}:continue
        poly=shapely.from_wkb(bytes.fromhex(record['wkb_hex']));inside|=shapely.contains_xy(poly,corners[...,0],corners[...,1])
    return inside.all((1,2)).astype(float),flags

def geometry_known_score(features,known,valid):
    gap=features[...,64:120].reshape(*valid.shape,8,7)[...,0].min(-1)*10
    s=(5/14)*known[...,7]*known[...,8]-.25*(gap<=0)
    s-=known[...,3]*(1-known[...,4])+known[...,5]*(1-known[...,6])
    s+=(2/14)*known[...,9]
    refs=valid.argmax(1);return s-s[np.arange(len(s)),refs,None]

def progress_geometry(known):
    result=np.zeros((*known.shape[:-1],18),np.float32);result[...,0]=known[...,7];result[...,4]=known[...,8]
    return result
