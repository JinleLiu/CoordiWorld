"""Fixed route-seed blocks, dynamically admitted frozen methods, append-only attempts."""
import os,json,time,datetime,subprocess,hashlib,signal,shutil,fcntl,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path('<CW_WORKSPACE>');B=ROOT/'twentyeighth_20261031';OLD=ROOT/'twentyseventh_20261030';BENCH=OLD/'external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956';CARLA=OLD/'external/CARLA_0.9.15';CLIENT=OLD/'external/carla-client/bin/python'
Q=Path(os.environ.get('R18_RUN_ROOT',str(B/'bench2drive')))
XML=Path(os.environ.get('R18_ROUTE_XML',str(BENCH/'leaderboard/data/drivetransformer_bench2drive_dev10.xml')))
XML_SHA=os.environ.get('R18_XML_SHA','6dc398c3835578f026d6268c9a8db4bcabf2113a2f9a141611b66fe48bc68327')
ONLY_METHOD=os.environ.get('R18_ONLY_METHOD');ONLY_ROUTE=os.environ.get('R18_ONLY_ROUTE');SINGLE=os.environ.get('R18_SINGLE_JOB')=='1'
WORKER=os.environ.get('R18_QUEUE_WORKER','all');PORT=int(os.environ.get('R18_RPC_PORT','29318'));TM_PORT=int(os.environ.get('R18_TM_PORT','49318'))
NATIVE_PUBLIC={'BasicAgent25','World4Scorer_route_tp','TFpp_single'}
TRANSFER={'World4Scorer_EC','World4Scorer_EC_H3','World4Scorer_EC_Gspeed'}
GROUP={'structured':{'CoordiWorld_Full','CoordiWorld_BaseEC'},'full':{'CoordiWorld_Full'},'base':{'CoordiWorld_BaseEC'},'public':NATIVE_PUBLIC|TRANSFER,'public_core_w4':{'World4Scorer_route_tp'},'public_core_other':{'BasicAgent25','TFpp_single'},'public_transfer':TRANSFER,'transfer_ec':{'World4Scorer_EC'},'transfer_h3':{'World4Scorer_EC_H3'},'transfer_gspeed':{'World4Scorer_EC_Gspeed'}}
STATUS=Q/('QUEUE_STATUS.json'if WORKER=='all'else f'QUEUE_STATUS_{WORKER}.json')
PAUSE=Q/('PAUSE_QUEUE'if WORKER=='all'else f'PAUSE_{WORKER}')
QUEUE_SOURCE_BYTES=Path(__file__).read_bytes();QUEUE_SOURCE_SHA=hashlib.sha256(QUEUE_SOURCE_BYTES).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def run():
 routes=[r.attrib for r in ET.parse(XML).getroot().findall('route') if not ONLY_ROUTE or r.get('id')==ONLY_ROUTE];assert sha(XML)==XML_SHA
 reg=dict(created_utc=stamp(),cohort=os.environ.get('R18_COHORT','Dev10_exposed'),routes=routes,xml=str(XML),xml_sha=sha(XML),seeds=[17,23,42],repetitions=1,sim_tick=.05,timeout_simulator_rpc_seconds=600,episode_wall_cap_seconds=3600,infra_retries_before_first_control=1,policy_failure_score=0,method_order='rotate registered method order by route index plus seed index; new methods catch up fixed earlier blocks',score='official per-route result; mean over seeds within route, then route means/cluster bootstrap; no fixed220 denominator',R17_results_reused=False)
 if not(Q/'PROTOCOL.json').exists():write(Q/'PROTOCOL.json',reg)
 while True:
  if PAUSE.exists():
   if SINGLE:return
   time.sleep(5);continue
  methods=json.loads((Q/'methods.json').read_text())
  native_ready=[m['id']for m in methods if m['id']in NATIVE_PUBLIC and m['status']=='ready']
  native_complete=all((Q/'episodes/seed17'/r['id']/m/'FINAL.json').exists()for r in routes for m in native_ready)
  ready=[m for m in methods if m['status']=='ready'and(m['id']==ONLY_METHOD if ONLY_METHOD else WORKER=='all'or m['id']in GROUP[WORKER])and(m['id']not in TRANSFER or native_complete or os.environ.get('R18_NATIVE_DEV_COMPLETE')=='1')];chosen=None
  for si,seed in enumerate([17,23,42]):
   if seed not in json.loads((Q/'queue_control.json').read_text())['enabled_seeds']:continue
   for ri,r in enumerate(routes):
    order=ready[(ri+si)%len(ready):]+ready[:(ri+si)%len(ready)]if ready else[]
    for m in order:
     if seed not in m.get('eval_seeds',[17,23,42]):continue
     out=Q/'episodes'/f"seed{seed}"/r['id']/m['id']
     if not(out/'FINAL.json').exists():chosen=(seed,r,m,out);break
    if chosen:break
   if chosen:break
  if not chosen and SINGLE:return
  if not chosen:write(STATUS,dict(status='WAITING_READY_METHOD_OR_SEED',utc=stamp(),worker=WORKER));time.sleep(15);continue
  seed,r,m,out=chosen;out.mkdir(parents=True,exist_ok=True)
  for attempt in range(2):
   ad=out/f'attempt{attempt+1}';ad.mkdir(exist_ok=True)
   if (ad/'ATTEMPT.json').exists():
    prior=json.loads((ad/'ATTEMPT.json').read_text())
    if prior.get('terminal'):write(out/'FINAL.json',prior);break
    continue
   cfg=dict(m.get('config',{}),output_dir=str(ad/'trace'),eval_seed=seed,method_config_sha=m['config_sha']);write(ad/'agent.json',cfg)
   # Simulator must run as the existing nonroot CARLA user.
   subprocess.run(['chown','-R','cwcarla:cwcarla',str(out)],check=True)
   env=os.environ.copy();env.update(CARLA_ROOT=str(CARLA),LEADERBOARD_ROOT=str(BENCH/'leaderboard'),SCENARIO_RUNNER_ROOT=str(BENCH/'scenario_runner'),IS_BENCH2DRIVE='True',XDG_RUNTIME_DIR='/tmp/cwcarla-runtime',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   env['PYTHONPATH']=':'.join([str(B/'scripts'),str(OLD/'scripts'),str(OLD/'bench2drive/compat'),str(CARLA/'PythonAPI'),str(CARLA/'PythonAPI/carla'),str(BENCH/'leaderboard'),str(BENCH/'scenario_runner'),*m.get('pythonpath',[])])
   env.update(m.get('env',{}));cp=ad/'checkpoint.json'
   cmd=['runuser','-u','cwcarla','--',m.get('python',str(CLIENT)),str(BENCH/'leaderboard/leaderboard/leaderboard_evaluator.py'),'--routes',str(XML),'--routes-subset',r['id'],'--repetitions','1','--track',m['track'],'--checkpoint',str(cp),'--debug-checkpoint',str(ad/'live_results.txt'),'--agent',m['agent'],'--agent-config',str(ad/'agent.json'),'--port',str(PORT),'--traffic-manager-port',str(TM_PORT),'--traffic-manager-seed',str(seed),'--gpu-rank','0','--timeout','600']
   shutil.copy2(m['agent'],ad/'executed_agent_source.py');(ad/'executed_queue_source.py').write_bytes(QUEUE_SOURCE_BYTES)
   start=time.monotonic();receipt=dict(started_utc=stamp(),method=m['id'],seed=seed,route=r,config_sha=m['config_sha'],method_registration=m,agent_source_sha=sha(m['agent']),cmd=cmd,attempt=attempt+1,terminal=False)
   receipt.update(worker=WORKER,rpc_port=PORT,tm_port=TM_PORT,queue_source_sha=QUEUE_SOURCE_SHA,cohort=os.environ.get('R18_COHORT','Dev10_exposed'))
   write(ad/'START.json',receipt);write(STATUS,receipt|dict(status='RUNNING',out=str(ad)))
   with open(ad/'runner.log','w')as log:
    p=subprocess.Popen(cmd,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=p.wait(timeout=3600);timeout=False
    except subprocess.TimeoutExpired:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     rc=-9;timeout=True
   records=json.loads(cp.read_text()).get('_checkpoint',{}).get('records',[])if cp.exists()else[]
   traces=[json.loads(line)for f in (ad/'trace').rglob('trace.jsonl')for line in open(f)]if(ad/'trace').exists()else[];steps=len(traces)
   model_steps=sum(int(x.get('policy_replans',0))>0 for x in traces)if m['id'].startswith('CoordiWorld_')else steps
   record=records[-1]if records else None
   decision_started=model_steps>0
   terminal=bool(record and record.get('status')=='Completed')or decision_started or attempt==1
   receipt.update(ended_utc=stamp(),wall_seconds=time.monotonic()-start,returncode=rc,wall_timeout=timeout,control_steps_logged=steps,model_control_steps=model_steps,decision_started=decision_started,official_record=record,terminal=terminal,outcome='official_record'if record else'failed_missing_record',score_if_missing=0)
   write(ad/'ATTEMPT.json',receipt)
   with open(Q/('attempts.jsonl'if WORKER=='all'else f'attempts_{WORKER}.jsonl'),'a')as f:f.write(json.dumps(receipt)+'\n')
   if terminal:
    write(out/'FINAL.json',receipt)
    if not decision_started:
     # A systematic import/startup failure must not fan out over all routes.
     with open(Q/'methods.lock','a')as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      mr=json.loads((Q/'methods.json').read_text())
      for mm in mr:
       if mm['id']==m['id']:mm.update(status='blocked_infrastructure',blocked_attempt=str(ad),blocked_utc=stamp())
      write(Q/'methods.json',mr)
    break
  if SINGLE:return
if __name__=='__main__':run()
