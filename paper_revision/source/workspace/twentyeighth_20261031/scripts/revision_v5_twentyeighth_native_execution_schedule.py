"""Lease slots only after original methods complete; then release to coverage."""
import datetime,json,os,subprocess,time
from pathlib import Path
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');Q=B/'native_execution_run/bench2drive';P='<CW_WORKSPACE>/envs/v5/bin/python'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 proto=json.load(open(Q/'PROTOCOL.json'));routes=[r['id']for r in proto['routes']];jobs=[('transfer_h3','World4Scorer_EC_H3','native_h3','World4Scorer_NativeEC_H3',29718),('transfer_gspeed','World4Scorer_EC_Gspeed','native_gspeed','World4Scorer_NativeEC_Gspeed',29318),('transfer_ec','World4Scorer_EC','native_ec','World4Scorer_NativeEC',29518)];launched={}
 while True:
  for oldworker,oldmethod,worker,method,port in jobs:
   if worker in launched:continue
   expected=[B/'bench2drive/episodes'/f'seed{s}'/r/oldmethod/'FINAL.json'for s in [17,23,42]for r in routes]
   if not all(p.exists()for p in expected):continue
   (B/'bench2drive'/('PAUSE_'+oldworker)).write_text('Original30 complete; native-execution registered comparison takes this slot\n')
   env=os.environ.copy();env.update(R18_RUN_ROOT=str(Q),R18_ROUTE_XML=proto['xml'],R18_XML_SHA=proto['xml_sha'],R18_COHORT=proto['cohort'],R18_ONLY_METHOD=method,R18_NATIVE_DEV_COMPLETE='1',R18_QUEUE_WORKER=worker,R18_RPC_PORT=str(port),R18_TM_PORT=str(port+20000))
   with open(Q/f'{worker}.log','a')as f:p=subprocess.Popen([P,str(B/'scripts/queue_v09.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   launched[worker]=dict(pid=p.pid,method=method,port=port,utc=now());write(Q/'SLOT_RELEASES.json',launched)
  expected=[Q/'episodes/seed17'/r/m/'FINAL.json'for _,_,_,m,_ in jobs for r in routes];done=sum(p.exists()for p in expected);write(Q/'SCHEDULE_STATUS.json',dict(utc=now(),completed=done,expected=30,launched=launched))
  if done==30:
   for _,_,w,_,_ in jobs:(Q/('PAUSE_'+w)).write_text('Registered native execution30 complete\n')
   write(Q/'FINISHED.json',dict(utc=now(),completed=done,launched=launched));(Q/'ACTIVE').unlink(missing_ok=True);return
  time.sleep(30)
if __name__=='__main__':main()
