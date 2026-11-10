"""Prepare frozen static-map caches from installed OpenDRIVE, without simulation.

CARLA0.9.15 Map.cpp constructors use the same OpenDRIVE MakeMap backend.
No spawn-point API or live actors are consumed by export_map.
First require exact equality with the actual Town03 cache before new maps.
"""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
import carla
from revision_v5_twentyeighth_carla_view import export_map
B=Path('<CW_WORKSPACE>/twentyeighth_20261031')
C=B.parent/'twentyseventh_20261030/external/CARLA_0.9.15'
O=B/'bridge/static_map_preparation';O.mkdir(parents=True,exist_ok=True)
assets=json.load(open(B/'research/COVERAGE_MAP_ASSETS.json'));rows=[]
source=dict(url='https://github.com/carla-simulator/carla/blob/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/LibCarla/source/carla/client/Map.cpp',commit='d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18',file_blob_sha='3e775de5a8d16e58da7f47613db524c155f8f678',read_lines=[30,70],adaptation='Use identical client Map(name, OpenDRIVE) constructor for current static lanes only; no simulator world, traffic, route result or actor state.')
for item in sorted(assets,key=lambda x:(x['town']!='Town03',x['town'])):
    town=item['town'];assert len(item['opendrive'])==len(item['map_files'])==1
    raw=Path(item['opendrive'][0]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['opendrive'][0]['sha256']
    name=str(Path(item['map_files'][0]).relative_to('CarlaUE4/Content').with_suffix(''))
    start=time.monotonic();cmap=carla.Map(name,raw.decode());path=O/(town+'.json')
    data=export_map(cmap,path);target=B/'bridge/maps'/(town+'.json')
    if town=='Town03':assert data==json.load(open(target)),'Offline map differs from actual captured Town03; do not prepare other caches'
    if target.exists():assert data==json.load(open(target))
    else:
        tmp=target.with_suffix('.prepare-'+str(os.getpid()));tmp.write_bytes(path.read_bytes());tmp.replace(target)
    row=dict(town=town,name=name,lanes=len(data['lanes']),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),opendrive_sha256=data['opendrive_sha'],seconds=time.monotonic()-start,existing_live_Town03_exact=(town=='Town03'))
    rows.append(row);print(json.dumps(row),flush=True)
    (O/'RESULT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,source=source,rows=rows,scope='Static input preparation only, no new episode or control. Installed map assets all present. Existing Town03 data equality checked. New live-map names/OpenDRIVE identities still checked against recorded episode metadata.'),indent=2)+'\n')
