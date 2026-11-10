"""Run only on a reserved idle simulator, before confirmation policy execution."""
import datetime,hashlib,json,sys,time
from pathlib import Path
import carla
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');c=carla.Client('localhost',int(sys.argv[1]));c.set_timeout(180);out=[]
for town in ['Town06','Town02']:
 t=time.monotonic();w=c.load_world(town);m=w.get_map();live=hashlib.sha256(m.to_opendrive().encode()).hexdigest();p=B/'bridge/maps'/f'{town}.json';j=json.load(open(p));row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),town=town,live_name=m.name,live_opendrive_sha=live,cached_opendrive_sha=j['opendrive_sha'],cached_map_sha=hashlib.sha256(p.read_bytes()).hexdigest(),cached_path=str(p),equal=live==j['opendrive_sha'],seconds=time.monotonic()-t);out.append(row);(B/'research/confirmation_v03/LIVE_MAP_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');assert row['equal'],row
print(json.dumps(out,indent=2))
