"""Only the frozen 24-job development matrix. Never launches confirmation.

Separate runner processes/ports, bounded pre-control retry, all failures kept.
Global stop on interface/prefix failure; already-running jobs finish unchanged.
"""
from pathlib import Path
import os,json,sys,time,signal,hashlib,subprocess,fcntl,socket,datetime,traceback
ROOT=Path('<CW_WORKSPACE>');RUN=ROOT/'r18_followup_20261102T030813Z';Q=RUN/'development_v02';OLD=ROOT/'twentyeighth_20261031';B17=ROOT/'twentyseventh_20261030';BENCH=B17/'external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956';CARLA=B17/'external/CARLA_0.9.15'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def frozen():
    p=RUN/'protocol/DEVELOPMENT_REPAIR_FREEZE_v02.json';f=read(p);original=read(RUN/'protocol/DEVELOPMENT_FREEZE_v01.json')
    assert f['phase']=='DEVELOPMENT' and f['episode_budget']==24 and not f['confirmation_authorized_by_this_file']
    assert f['schedule']==original['schedule'] and f['slots']==original['slots'] and f['r_star_rule']==original['r_star_rule']
    for rel,h in f['file_sha256'].items():assert sha(RUN/rel)==h,('frozen source changed',rel)
    audit=read(RUN/'diagnostics/interface_repair_v02/RESULT.json');assert audit['passed']and not audit['errors']and audit['new_episodes']==0 and len(audit['results'])==8
    for r in audit['results']:
        assert r['reuse_eligible']and r['admission_equivalent']and not r['admission_changes']and r['corrected_selected_prefix_gate']['passed']
        assert sha(Path(r['episode'])/'FINAL.json')==r['original_final_sha256']
        assert r['source_prefix_sha256']==sha(RUN/f['code_directory']/'cw_controller_prefix.py')
    assert read(RUN/'development_v01/COORDINATOR_FINAL.json')['status']=='STOPPED_FOR_INTERFACE_OR_INFRASTRUCTURE'
    assert read(RUN/'diagnostics/common_interface_probe_final_v01/RESULT.json')['status']=='PASSED_COMMON_POOL_CONTRACT'
    assert read(RUN/'diagnostics/queue_safety_checks_v01/RESULT.json')['passed']
    for arm in (RUN/'diagnostics/common_interface_probe_final_v01').glob('*/*/ARM_RESULT.json'):
        for name,h in read(arm)['source_hashes'].items():
            if name!='cw_controller_prefix.py':assert sha(RUN/f['code_directory']/name)==h,('unchanged contract source mismatch',name)
    assert len(f['schedule'])==24 and len({(j['route'],j['method'],j['seed'])for j in f['schedule']})==24
    return f,sha(p)
def _server_cleanup(ad):
    p=ad/'OWNED_SERVER.json'
    if not p.exists():return dict(action='no_server_identity')
    r=read(p);pid=r['pid'];stat=Path(f'/proc/{pid}/stat')
    if not stat.exists():return dict(action='already_gone',pid=pid)
    fields=stat.read_text().split()
    if fields[21]!=r['birth_ticks']or os.getpgid(pid)!=pid:return dict(action='identity_mismatch_preserved',pid=pid)
    os.killpg(pid,signal.SIGTERM)
    for _ in range(30):
        if not stat.exists():return dict(action='owned_group_terminated',pid=pid)
        time.sleep(.1)
    if stat.exists()and stat.read_text().split()[21]==r['birth_ticks']:os.killpg(pid,signal.SIGKILL)
    return dict(action='owned_group_killed',pid=pid)
def server_cleanup(ad):
    try:return _server_cleanup(ad)
    except (ProcessLookupError,FileNotFoundError):return dict(action='already_gone_during_cleanup')
def worker(index,slot):
    f,freeze_sha=frozen();job=f['schedule'][index];method=f['methods'][job['method']];port=f['slots'][slot]['rpc'];tm=f['slots'][slot]['tm'];guard=lock(Q/'locks'/f'slot_{port}.lock');jobguard=lock(Q/'locks'/f'job_{index:02d}.lock');out=Q/'episodes'/f"seed{job['seed']}"/job['route']/method['id'];out.mkdir(parents=True,exist_ok=True)
    if(out/'FINAL.json').exists():assert read(out/'FINAL.json')['freeze_sha256']==freeze_sha;return
    for attempt in [1,2]:
        ad=out/f'attempt{attempt}';ad.mkdir(exist_ok=True)
        if(ad/'START.json').exists():raise RuntimeError('Unfinished prior attempt requires explicit recovery accounting; never overwrite or silently rerun')
        for p in [port,port+1,port+2,tm]:
            s=socket.socket();occupied=s.connect_ex(('127.0.0.1',p))==0;s.close();assert not occupied,('occupied_slot',p)
        config={**method['config'],'output_dir':str(ad/'trace'),'eval_seed':job['seed'],'method_config_sha':method['config_sha256']};write(ad/'agent.json',config);subprocess.run(['chown','-R','cwcarla:cwcarla',str(out)],check=True)
        env={**os.environ,'CARLA_ROOT':str(CARLA),'LEADERBOARD_ROOT':str(BENCH/'leaderboard'),'SCENARIO_RUNNER_ROOT':str(BENCH/'scenario_runner'),'IS_BENCH2DRIVE':'True','XDG_RUNTIME_DIR':'/tmp/cwcarla-runtime','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','CW_OWNED_SERVER_FILE':str(ad/'OWNED_SERVER.json')}
        env['PYTHONPATH']=':'.join([str(RUN/f['code_directory']),str(OLD/'scripts'),str(B17/'scripts'),str(B17/'bench2drive/compat'),str(CARLA/'PythonAPI'),str(CARLA/'PythonAPI/carla'),str(BENCH/'leaderboard'),str(BENCH/'scenario_runner')])
        cp=ad/'checkpoint.json';cmd=['runuser','-u','cwcarla','--',method['python'],str(RUN/f['code_directory']/'owned_leaderboard_evaluator.py'),'--routes',str(RUN/f['route_xml']),'--routes-subset',job['route'],'--repetitions','1','--track','MAP','--checkpoint',str(cp),'--debug-checkpoint',str(ad/'live_results.txt'),'--agent',str(RUN/f['code_directory']/'cw_followup_agent.py'),'--agent-config',str(ad/'agent.json'),'--port',str(port),'--traffic-manager-port',str(tm),'--traffic-manager-seed',str(job['seed']),'--gpu-rank','0','--timeout','600']
        rec=dict(started_utc=now(),freeze_sha256=freeze_sha,job=job,method=method['id'],seed=job['seed'],route=job['route'],slot=slot,rpc_port=port,tm_port=tm,cohort='R18_followup_v02_development_exposed6',attempt=attempt,config_sha256=sha(ad/'agent.json'),template_config_sha256=method['config_sha256'],code_directory=f['code_directory'],source_hashes=f['file_sha256'],command=cmd,terminal=False)
        write(ad/'START.json',rec);begin=time.monotonic()
        with (ad/'runner.log').open('w')as log:
            p=subprocess.Popen(cmd,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(ad/'RUNNER_PID.json',dict(pid=p.pid,birth_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]))
            timed_out=False
            try:rc=p.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                rc=p.returncode
        clean=server_cleanup(ad);records=[];parse_errors=[]
        if cp.exists():
            try:records=read(cp).get('_checkpoint',{}).get('records',[])
            except Exception as e:parse_errors.append(repr(e))
        traces=[];tp=ad/'trace/trace.jsonl'
        if tp.exists():
            for line in tp.read_text().splitlines():
                try:traces.append(json.loads(line))
                except ValueError:parse_errors.append('truncated trace row preserved')
        model_steps=sum(x.get('policy_replans',0)>0 for x in traces);started=model_steps>0;record=records[-1]if records else None;terminal=bool(record and record.get('status')=='Completed')or started or attempt==2;prefix=None
        if started:
            try:
                from validate_actual_episode_prefix import validate
                prefix=validate(ad/'trace');write(ad/'ACTUAL_PREFIX_VALIDATION.json',prefix)
            except Exception:prefix=dict(passed=False,error=traceback.format_exc());write(ad/'ACTUAL_PREFIX_VALIDATION.json',prefix)
        rec.update(ended_utc=now(),wall_seconds=time.monotonic()-begin,returncode=rc,wall_timeout=timed_out,control_steps_logged=len(traces),model_control_steps=model_steps,decision_started=started,official_record=record,terminal=terminal,score_if_missing=0,missing_record_events='UNKNOWN'if record is None else None,parse_errors=parse_errors,owned_server_cleanup=clean,prefix_validation=None if prefix is None else {k:v for k,v in prefix.items()if k!='rows'})
        write(ad/'ATTEMPT.json',rec)
        with (Q/'attempts.jsonl').open('a')as log:fcntl.flock(log,fcntl.LOCK_EX);log.write(json.dumps(rec)+'\n');log.flush()
        if terminal:
            write(out/'FINAL.json',rec)
            gate=bool(started and prefix and prefix['passed']and not parse_errors)
            write(Q/'jobs'/f'{index:02d}.json',dict(utc=now(),job=job,final=str(out/'FINAL.json'),gate_passed=gate,decision_started=started,official_record_present=record is not None,prefix_validation=rec['prefix_validation'],freeze_sha256=freeze_sha))
            if not gate:write(Q/'INTERFACE_STOP.json',dict(utc=now(),job=job,reason='Control/input/prefix gate failed or no decision started; preserve outcomes and inspect before more dispatch',final=str(out/'FINAL.json')))
            return
        time.sleep(2)
def coordinator():
    guard=lock(Q/'coordinator.lock');f,freeze_sha=frozen();running={};logs={};done={};slots=range(3)
    for i in range(24):
        p=Q/'jobs'/f'{i:02d}.json'
        if p.exists():done[i]=read(p);assert done[i]['freeze_sha256']==freeze_sha
    while len(done)<24 or running:
        for slot,(index,p)in list(running.items()):
            if p.poll()is not None:
                logs.pop(slot).close();running.pop(slot);jp=Q/'jobs'/f'{index:02d}.json'
                if p.returncode or not jp.exists():write(Q/'INTERFACE_STOP.json',dict(utc=now(),reason='worker terminated before accounted FINAL',job_index=index,returncode=p.returncode));done[index]=dict(gate_passed=False,worker_error=True)
                else:done[index]=read(jp)
        stop=(Q/'INTERFACE_STOP.json').exists()or(Q/'PAUSE_NEW_DISPATCH').exists()
        if not stop:
            for slot in slots:
                if slot in running:continue
                pending=[i for i in range(24)if i not in done and i not in [j for j,_ in running.values()]]
                if not pending:break
                fs=os.statvfs(RUN)
                if fs.f_bavail*fs.f_frsize<6*1024**3:write(Q/'PAUSE_NEW_DISPATCH',dict(utc=now(),reason='server disk below6GiB; preserve assets, inspect storage'));stop=True;break
                i=pending[0];lp=Q/'logs'/f'worker_{i:02d}.log';lp.parent.mkdir(parents=True,exist_ok=True);log=lp.open('a');p=subprocess.Popen([sys.executable,__file__,'worker',str(i),str(slot)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[slot]=(i,p);logs[slot]=log
        status=dict(utc=now(),pid=os.getpid(),phase='DEVELOPMENT',freeze_sha256=freeze_sha,completed_jobs=len(done),expected_jobs=24,remaining_jobs=24-len(done),active=[dict(slot=s,index=i,pid=p.pid,job=f['schedule'][i])for s,(i,p)in running.items()],dispatch_stopped=stop,confirmation_started=False,additional_budget_spent=0)
        write(Q/'QUEUE_STATUS.json',status)
        if stop and not running:break
        if len(done)==24 and not running:break
        time.sleep(10)
    write(Q/'COORDINATOR_FINAL.json',dict(utc=now(),status='DEVELOPMENT_MATRIX_COMPLETE'if len(done)==24 and all(x.get('gate_passed')for x in done.values())else'STOPPED_FOR_INTERFACE_OR_INFRASTRUCTURE',completed_jobs=len(done),confirmation_started=False,requires_development_analysis_and_separate_confirmation_freeze=True))
if __name__=='__main__':
    Q.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    else:coordinator()
