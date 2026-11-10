"""Fixed three-block window on three existing slots; resume existing jobs unchanged."""
import datetime,json,os,subprocess,time
from pathlib import Path
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'research/confirmation_v03';P=str(B.parent/'envs/v5/bin/python');schedule=json.load(open(O/'SCHEDULE.json'));blocks=schedule['blocks'];freeze=json.load(open(O/'FREEZE.json'));assert freeze['status']=='FROZEN_FOR_CONFIRMATION';assert all(r['equal']for r in json.load(open(O/'LIVE_MAP_CHECK.json')));rep=json.load(open(B/'research/supply_replication/STATUS.json'));assert not rep['pending']and not rep['active'];assert len(rep['done'])==6
assert json.load(open(O/'MAP_CHECK_COMPLETE.json'))['server_stopped']
cutoff=datetime.datetime.fromisoformat(json.load(open(B/'RUN.json'))['stable_queue_extension_deadline_utc'])-datetime.timedelta(seconds=3615);active={};attempted=set();finished=[];stopped=False
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def final(b,m):return Path(b['root'])/'episodes'/f"seed{b['seed']}"/b['route']/m/'FINAL.json'
class ResumedProcess:
 def __init__(self,rec):self.pid=rec['pid'];self.start_ticks=rec.get('process_start_ticks');self.returncode=None
 def poll(self):
  try:
   fields=Path(f'/proc/{self.pid}/stat').read_text().split()
   if fields[2]=='Z'or(self.start_ticks is not None and fields[21]!=self.start_ticks):return 0
   return None
  except FileNotFoundError:return 0
resume=os.environ.get('R18_CONFIRM_RESUME_FROM')
if resume:
 receipt=json.load(open(resume));old=receipt['prior_status'];assert not old['stopped_new_jobs'];finished=old['terminal_processes']
 for r in old['active']:
  b=blocks[r['block']];assert b['route']==r['route'];active[r['port']]=(ResumedProcess(r),b,r['method']);attempted.add((r['block'],r['method']))
 for r in finished:attempted.add((r['block'],r['method']))
while True:
 for port,(p,b,m)in list(active.items()):
  if p.poll()is not None:
   finished.append(dict(block=b['block_index'],method=m,returncode=p.returncode,terminal=final(b,m).exists(),resumed_process=isinstance(p,ResumedProcess),exit_code_note='Coordinator exit code unavailable after adoption; actual runner return code retained in ATTEMPT/FINAL'if isinstance(p,ResumedProcess)else None));del active[port]
   if not final(b,m).exists():stopped=True
   elif not json.load(open(final(b,m))).get('decision_started'):stopped=True
 done=[b['block_index']for b in blocks if all(final(b,m).exists()for m in b['methods'])];pending=[b for b in blocks if b['block_index']not in done];window=pending[:3]
 now=datetime.datetime.now(datetime.timezone.utc)
 if now>=cutoff:stopped=True
 if not stopped:
  for port in [29318,29518,29718]:
   if port in active:continue
   if port==29718 and not(B/'tight_ablation_run/bench2drive/episodes/seed17/3514/CoordiWorld_MapTight_NoWitness_v01/FINAL.json').exists():continue
   job=next(((b,m)for b in window for m in b['methods']if not final(b,m).exists()and(b['block_index'],m)not in attempted),None)
   if job is None:continue
   b,m=job;q=Path(b['root']);proto=json.load(open(q/'PROTOCOL.json'));env=os.environ.copy();env.update(R18_RUN_ROOT=str(q),R18_ROUTE_XML=proto['xml'],R18_XML_SHA=proto['xml_sha'],R18_COHORT=proto['cohort'],R18_ONLY_METHOD=m,R18_ONLY_ROUTE=b['route'],R18_NATIVE_DEV_COMPLETE='1',R18_SINGLE_JOB='1',R18_QUEUE_WORKER='confirm_slot'+str([29318,29518,29718].index(port)),R18_RPC_PORT=str(port),R18_TM_PORT=str(port+20000))
   with open(O/f"block{b['block_index']:02d}_{m}.log",'a')as f:p=subprocess.Popen([P,str(B/'scripts/queue_v09.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   active[port]=(p,b,m);attempted.add((b['block_index'],m))
 status=dict(utc=now.isoformat(),coordinator_version='v02_three_block_window',pending_block_window=3,complete_blocks=done,total_blocks=len(blocks),active=[dict(port=port,pid=p.pid,block=b['block_index'],route=b['route'],cohort=b['cohort'],method=m)for port,(p,b,m)in active.items()],terminal_processes=finished,stopped_new_jobs=stopped,stop_rule='Infrastructure interruption or original35h reserve cutoff; outcome-independent fixed prefix.');write(O/'STATUS.json',status)
 if not active and(stopped or not pending):break
 time.sleep(15)
