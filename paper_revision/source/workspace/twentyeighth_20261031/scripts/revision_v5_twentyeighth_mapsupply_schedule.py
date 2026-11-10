"""Finish original W4 blocks; launch matched map pilot on the released slots."""
import datetime,json,os,subprocess,time
from pathlib import Path
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');Q=B/'mapsupply_pilot_run/bench2drive';P=str(B.parent/'envs/v5/bin/python');launched={}
items=[('World4Scorer_EC_Gspeed','transfer_gspeed','CoordiWorld_BaseEC_v05',29318),('World4Scorer_EC_H3','transfer_h3','CoordiWorld_BaseEC_MapSupply_v04',29718)]
while True:
 if (Q/'PAUSE_SCHEDULE').exists():break
 for old,worker,mid,port in items:
  if mid in launched:continue
  if len(list((B/'bench2drive/episodes').glob(f'seed*/*/{old}/FINAL.json')))!=30:continue
  (B/'bench2drive'/f'PAUSE_{worker}').write_text('Completed original30; port reallocated to R18v03 spatial pilot\n')
  proto=json.load(open(Q/'PROTOCOL.json'));env=os.environ.copy();env.update(R18_RUN_ROOT=str(Q),R18_ROUTE_XML=proto['xml'],R18_XML_SHA=proto['xml_sha'],R18_COHORT=proto['cohort'],R18_ONLY_METHOD=mid,R18_NATIVE_DEV_COMPLETE='1',R18_QUEUE_WORKER='map_'+worker,R18_RPC_PORT=str(port),R18_TM_PORT=str(port+20000))
  with open(Q/(mid+'.log'),'a')as f:p=subprocess.Popen([P,str(B/'scripts/queue_v09.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  launched[mid]=dict(pid=p.pid,port=port,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 n=len(list((Q/'episodes').glob('seed*/*/*/FINAL.json')));(Q/'SCHEDULE_STATUS.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launched=launched,completed=n,expected=4),indent=2)+'\n')
 if n==4:
  for _,worker,_,_ in items:(Q/f'PAUSE_map_{worker}').write_text('All registered spatial pilot episodes complete\n')
  break
 time.sleep(20)
