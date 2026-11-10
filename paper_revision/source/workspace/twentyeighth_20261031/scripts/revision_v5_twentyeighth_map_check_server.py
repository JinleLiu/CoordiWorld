"""Temporary unscored map-only CARLA server on a released episode slot."""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
import carla
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'research/confirmation_v03';state=json.load(open(B/'research/supply_replication/STATUS.json'));assert not state['pending'];assert all(x['port']!=29318 for x in state['active']);env=os.environ.copy();env['XDG_RUNTIME_DIR']='/tmp/cwcarla-runtime';cmd=['runuser','-u','cwcarla','--',str(B.parent/'twentyseventh_20261030/external/CARLA_0.9.15/CarlaUE4.sh'),'/Game/Carla/Maps/Town06','-RenderOffScreen','-nosound','-carla-rpc-port=29318','-graphicsadapter=0'];log=open(O/'map_server.log','w');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rc=None
try:
 for _ in range(30):
  try:c=carla.Client('127.0.0.1',29318);c.set_timeout(3);c.get_world();break
  except RuntimeError:time.sleep(1)
 else:raise RuntimeError('Map-only server did not become ready')
 rc=subprocess.call(['bash',str(B/'scripts/cw_bridge_python.sh'),str(B/'scripts/revision_v5_twentyeighth_live_map_check.py'),'29318'],env=env)
finally:
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 log.close();(O/'MAP_CHECK_COMPLETE.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=rc,server_pid=p.pid,server_stopped=True,scope='Unscored static map check only; prior no-server timeout preserved. Benchmark evaluator normally starts/stops its own server per episode.'),indent=2)+'\n')
assert rc==0,rc
