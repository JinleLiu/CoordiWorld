"""One frozen historical-platform phase. No automatic second matrix or retries."""
from pathlib import Path
import os,json,sys,time,signal,hashlib,subprocess,fcntl,socket,datetime,traceback
R=Path('<CW_WORKSPACE>');RUN=R/'r18_followup_20261102T030813Z';V=RUN/'core_contribution_v05';OLD=R/'twentyeighth_20261031';B17=R/'twentyseventh_20261030';BENCH=B17/'external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956';CARLA=B17/'external/CARLA_0.9.15'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def lock(p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def birth(pid):return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def frozen(phase):
 assert phase in ['bridge','confirmation'];p=V/'protocol'/f'{phase.upper()}_FREEZE_v01.json';f=read(p);assert f['phase']==phase
 assert f['episode_budget']==len(f['schedule']) and f['episode_budget']<=(6 if phase=='bridge'else 72)
 assert len({(j['route'],j['seed'],j['method'])for j in f['schedule']})==len(f['schedule'])
 for rel,h in f['file_sha256'].items():assert sha(V/rel)==h,('frozen_file_changed',rel)
 for path,h in f['external_source_sha256'].items():assert sha(path)==h,('historical_source_changed',path)
 for path,fp in f['model_fingerprints'].items():
  s=Path(path).stat();assert dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)==fp,('model_changed',path)
 if phase=='confirmation':
  receipt=read(V/'runtime/BRIDGE_ACCEPTANCE.json');assert receipt['passed'] and receipt['driving_attempts']<=6
 return f,sha(p)
def cleanup(ad):
 p=ad/'OWNED_SERVER.json'
 if not p.exists():return {'action':'no_owned_server'}
 r=read(p);pid=r['pid'];s=Path(f'/proc/{pid}/stat')
 try:
  if not s.exists():return {'action':'already_exited'}
  if birth(pid)!=r['birth_ticks']or os.getpgid(pid)!=pid:return {'action':'identity_mismatch_preserved'}
  os.killpg(pid,signal.SIGTERM)
  for _ in range(30):
   if not s.exists():return {'action':'owned_group_terminated'}
   time.sleep(.1)
  if s.exists()and birth(pid)==r['birth_ticks']:os.killpg(pid,signal.SIGKILL)
  return {'action':'owned_group_killed_after_grace'}
 except ProcessLookupError:return {'action':'already_exited'}
def worker(phase,index,slot):
 f,fh=frozen(phase);q=V/(phase+'_v01');job=f['schedule'][index];assert job['slot']==slot;m=f['methods'][job['method']];ports=f['slots'][slot];locks=[lock(q/'locks'/f'slot{slot}.lock'),lock(q/'locks'/f'job{index}.lock')]
 ep=q/'episodes'/f"seed{job['seed']}"/job['route']/job['method'];ad=ep/'attempt1'
 assert not ep.exists(),'An existing attempt/FINAL must never be overwritten or retried automatically'
 ad.mkdir(parents=True)
 for port in [ports['rpc'],ports['rpc']+1,ports['rpc']+2,ports['tm']]:
  with socket.socket()as s:assert s.connect_ex(('127.0.0.1',port))!=0,('port_occupied',port)
 write(ad/'agent.json',dict(m['config'],output_dir=str(ad/'trace'),eval_seed=job['seed'],method_config_sha=m['config_sha256']))
 subprocess.run(['chown','-R','cwcarla:cwcarla',str(ep)],check=True)
 env={**os.environ,'CARLA_ROOT':str(CARLA),'LEADERBOARD_ROOT':str(BENCH/'leaderboard'),'SCENARIO_RUNNER_ROOT':str(BENCH/'scenario_runner'),'IS_BENCH2DRIVE':'True','XDG_RUNTIME_DIR':'/tmp/cwcarla-runtime','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','CW_OWNED_SERVER_FILE':str(ad/'OWNED_SERVER.json')}
 cmd=['runuser','-u','cwcarla','--',m['python'],str(V/'code/owned_leaderboard_evaluator.py'),'--routes',str(V/f['route_xml']),'--routes-subset',job['route'],'--repetitions','1','--track','MAP','--checkpoint',str(ad/'checkpoint.json'),'--debug-checkpoint',str(ad/'live_results.txt'),'--agent',str(V/'code'/m['agent']),'--agent-config',str(ad/'agent.json'),'--port',str(ports['rpc']),'--traffic-manager-port',str(ports['tm']),'--traffic-manager-seed',str(job['seed']),'--gpu-rank','0','--timeout','600']
 rec=dict(started_utc=now(),freeze_sha256=fh,phase=phase,job=job,attempt=1,method=job['method'],historical_id=m['historical_id'],command=cmd,config_sha256=sha(ad/'agent.json'),cohort=f['cohort'],predecision_retries=0,postdecision_retries=0)
 write(ad/'START.json',rec);start=time.monotonic()
 with(ad/'runner.log').open('w')as log:
  p=subprocess.Popen(cmd,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(ad/'RUNNER_PID.json',dict(pid=p.pid,birth_ticks=birth(p.pid)))
  timed=False
  try:rc=p.wait(timeout=f['max_episode_wall_seconds'])
  except subprocess.TimeoutExpired:
   timed=True;os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc=p.returncode
 clean=cleanup(ad);records=[];errors=[];steps=0
 try:
  if(ad/'checkpoint.json').exists():records=read(ad/'checkpoint.json').get('_checkpoint',{}).get('records',[])
  if(ad/'trace/trace.jsonl').exists():steps=sum(json.loads(s).get('policy_replans',0)>0 for s in(ad/'trace/trace.jsonl').read_text().splitlines())
 except Exception:errors.append(traceback.format_exc())
 if len(records)>1:errors.append('Multiple official records in single cell')
 validation=None
 if steps:
  vr=subprocess.run([m['python'],str(V/'code/validate_v05_episode.py'),str(ad/'trace')],capture_output=True,text=True,timeout=300)
  (ad/'validation.log').write_text(vr.stdout+'\n'+vr.stderr)
  if(ad/'V05_INTERFACE_VALIDATION.json').exists():validation=read(ad/'V05_INTERFACE_VALIDATION.json')
  if vr.returncode:errors.append('INTERFACE_VALIDATION_FAILED: '+vr.stderr[-2000:])
 rec.update(ended_utc=now(),wall_seconds=time.monotonic()-start,returncode=rc,wall_timeout=timed,decision_started=bool(steps),model_control_steps=steps,official_record=records[-1]if records else None,score_if_missing=0,events_if_missing='UNKNOWN',terminal=True,parse_or_interface_errors=errors,owned_server_cleanup=clean,interface_validation=None if validation is None else {k:v for k,v in validation.items()if k!='rows'})
 write(ad/'ATTEMPT.json',rec);write(ep/'FINAL.json',rec)
 gate=bool(steps and validation and validation['passed']and not errors)
 write(q/'jobs'/f'{index:03d}.json',dict(utc=now(),job=job,freeze_sha256=fh,final=str(ep/'FINAL.json'),gate_passed=gate,decision_started=bool(steps),official_record_present=bool(records),wall_seconds=rec['wall_seconds']))
 if not gate:write(q/'INTERFACE_STOP.json',dict(utc=now(),job=job,reason='interface/init failure: preserve all running cells; stop further dispatch',errors=errors))
def coordinator(phase):
 f,fh=frozen(phase);q=V/(phase+'_v01');locks=[lock(RUN/'runtime/SCIENTIFIC_EXECUTION.lock'),lock(q/'coordinator.lock')]
 assert not(q/'COORDINATOR_FINAL.json').exists(),'Already-final phase must not restart'
 running={};logs={};done={}
 for p in(q/'jobs').glob('*.json'):d=read(p);assert d['freeze_sha256']==fh;done[d['job']['index']]=d
 write(V/'runtime/ACTIVE.json',dict(phase=phase,queue=str(q),pid=os.getpid(),birth_ticks=birth(os.getpid()),freeze_sha256=fh,started_utc=now()))
 while True:
  for slot,(j,p)in list(running.items()):
   if p.poll()is None:continue
   logs.pop(slot).close();running.pop(slot);jp=q/'jobs'/f"{j['index']:03d}.json"
   if p.returncode or not jp.exists():
    write(q/'INTERFACE_STOP.json',dict(utc=now(),job=j,reason='worker ended before accounted result',returncode=p.returncode));done[j['index']]={'gate_passed':False,'unaccounted_failure':True}
   else:done[j['index']]=read(jp)
  stopped=(q/'INTERFACE_STOP.json').exists()or(q/'PAUSE_NEW_DISPATCH').exists()
  if not stopped:
   for slot in range(3):
    if slot in running:continue
    pending=[j for j in f['schedule']if j['slot']==slot and j['index']not in done and j['index']not in [r['index']for r,_ in running.values()]]
    if not pending:continue
    st=os.statvfs(V)
    if st.f_bavail*st.f_frsize<8*1024**3:write(q/'PAUSE_NEW_DISPATCH',dict(utc=now(),reason='below8GiB reserve'));stopped=True;break
    j=pending[0];ep=q/'episodes'/f"seed{j['seed']}"/j['route']/j['method']
    assert not ep.exists(),'Unaccounted prior attempt requires explicit recovery; no relaunch'
    lp=q/'logs'/f"worker_{j['index']:03d}.log";lp.parent.mkdir(parents=True,exist_ok=True);log=lp.open('a');p=subprocess.Popen([sys.executable,__file__,phase,'worker',str(j['index']),str(slot)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[slot]=(j,p);logs[slot]=log
    deadline=time.monotonic()+120
    while not(ep/'attempt1/START.json').exists()and p.poll()is None and time.monotonic()<deadline:time.sleep(.05)
    if not(ep/'attempt1/START.json').exists():write(q/'INTERFACE_STOP.json',dict(utc=now(),job=j,reason='bootstrap failed or timeout; active process preserved'));stopped=True;break
  write(q/'QUEUE_STATUS.json',dict(utc=now(),phase=phase,freeze_sha256=fh,completed_jobs=len(done),expected_jobs=len(f['schedule']),remaining_jobs=len(f['schedule'])-len(done),active=[dict(slot=s,job=j,pid=p.pid,birth_ticks=birth(p.pid))for s,(j,p)in running.items()if p.poll()is None],dispatch_stopped=stopped))
  if not running and(stopped or len(done)==len(f['schedule'])):break
  time.sleep(10)
 write(q/'COORDINATOR_FINAL.json',dict(utc=now(),phase=phase,freeze_sha256=fh,status='MATRIX_COMPLETE'if len(done)==len(f['schedule'])and all(x['gate_passed']for x in done.values())else'STOPPED_FOR_INTERFACE_OR_INFRASTRUCTURE',completed_jobs=len(done),expected_jobs=len(f['schedule']),next_phase_automatically_dispatched=False))
if __name__=='__main__':
 phase=sys.argv[1]
 if len(sys.argv)>2 and sys.argv[2]=='worker':worker(phase,int(sys.argv[3]),int(sys.argv[4]))
 else:coordinator(phase)
