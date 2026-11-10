"""Runtime refuses all dispatch without the separate completed-development freeze.

One fixed 216-cell queue. No policy tuning, automatic second queue or resume of an
unaccounted attempt. Internal and original public SENSORS agents retain their own controllers.
"""
from pathlib import Path
import os,json,sys,time,signal,subprocess,fcntl,socket,datetime,traceback
from confirmation_contract import read,sha,validate_matrix,decision_steps,terminal_attempt,native_trace_validation,native_identity_validation,internal_identity_validation,next_job,physical_output
ROOT=Path('<CW_WORKSPACE>');RUN=ROOT/'r18_followup_20261102T030813Z';Q=RUN/'confirmation_v01';OLD=ROOT/'twentyeighth_20261031';B17=ROOT/'twentyseventh_20261030';BENCH=B17/'external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956';CARLA=B17/'external/CARLA_0.9.15'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def lock(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def frozen(validation_path=None):
    # The optional path is for read-only prepublication validation. Every actual
    # worker/coordinator calls the fixed default; no CLI override exists.
    p=Path(validation_path)if validation_path is not None else RUN/'protocol/CONFIRMATION_FREEZE_v01.json';f=read(p);validate_matrix(f)
    for phase,count,status in [('development_v02',24,'DEVELOPMENT_MATRIX_COMPLETE'),('matched_development_v01',12,'MATCHED_DEVELOPMENT_MATRIX_COMPLETE')]:
        final=read(RUN/phase/'COORDINATOR_FINAL.json');assert final['status']==status and final['completed_jobs']==count
        jobs=list((RUN/phase/'jobs').glob('*.json'));assert len(jobs)==count and all(read(x)['gate_passed']for x in jobs)
        assert not read(RUN/phase/'QUEUE_STATUS.json')['active']
    for rel,h in f['file_sha256'].items():assert sha(RUN/rel)==h,('frozen source changed',rel)
    for path,h in f['external_file_sha256'].items():assert sha(path)==h,('external source changed',path)
    for path,old in f['model_fingerprints'].items():
        st=Path(path).stat();assert dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)==old,('model file changed since prefreeze SHA verification',path)
    selection=read(RUN/f['r_star_receipt']);assert selection['r_star']==f['r_star']and selection['final']and selection['all_required_development_valid']
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
def build_launch(f,job,method,ad,slot):
    ports=f['slots'][slot]
    env={**os.environ,'CARLA_ROOT':str(CARLA),'LEADERBOARD_ROOT':str(BENCH/'leaderboard'),'SCENARIO_RUNNER_ROOT':str(BENCH/'scenario_runner'),'IS_BENCH2DRIVE':'True','XDG_RUNTIME_DIR':'/tmp/cwcarla-runtime','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','CW_OWNED_SERVER_FILE':str(ad/'OWNED_SERVER.json'),**method.get('env',{})}
    env['PYTHONPATH']=':'.join([str(RUN/f['code_directory']),str(OLD/'scripts'),str(B17/'scripts'),str(B17/'bench2drive/compat'),str(CARLA/'PythonAPI'),str(CARLA/'PythonAPI/carla'),str(BENCH/'leaderboard'),str(BENCH/'scenario_runner')]+method.get('pythonpath',[]))
    cmd=['runuser','-u','cwcarla','--',method['python'],str(RUN/f['code_directory']/'owned_leaderboard_evaluator.py'),'--routes',str(RUN/f['route_xml']),'--routes-subset',job['route'],'--repetitions','1','--track',method['track'],'--checkpoint',str(ad/'checkpoint.json'),'--debug-checkpoint',str(ad/'live_results.txt'),'--agent',method['agent'],'--agent-config',str(ad/'agent.json'),'--port',str(ports['rpc']),'--traffic-manager-port',str(ports['tm']),'--traffic-manager-seed',str(job['seed']),'--gpu-rank','0','--timeout',str(f['failures']['RPC_timeout_seconds'])]
    return cmd,env
def output_directory(f,job,method):
    out=physical_output(job,method,f['storage_roots']);out.mkdir(parents=True,exist_ok=True)
    logical=Q/'episodes'/f"seed{job['seed']}"/job['route'];logical.parent.mkdir(parents=True,exist_ok=True)
    with lock(Q/'locks'/f"block_{job['seed']}_{job['route']}.lock"):
        if not logical.is_symlink()and not logical.exists():logical.symlink_to(out.parent,target_is_directory=True)
        assert logical.is_symlink()and logical.resolve()==out.parent.resolve(),('unexpected logical output location',str(logical))
    return out,logical/method['id']
def worker(index,slot):
    f,freeze_sha=frozen();job=f['schedule'][index];assert job['slot']==slot;method=f['methods'][job['method']];port=f['slots'][slot]['rpc'];tm=f['slots'][slot]['tm']
    guard=lock(Q/'locks'/f'slot_{port}.lock');jobguard=lock(Q/'locks'/f'job_{index:03d}.lock');out,logical=output_directory(f,job,method)
    if(out/'FINAL.json').exists():
        assert read(out/'FINAL.json')['freeze_sha256']==freeze_sha
        assert(Q/'jobs'/f'{index:03d}.json').exists(),'FINAL without job receipt requires explicit accounting recovery'
        return
    for attempt in [1,2]:
        ad=out/f'attempt{attempt}';ad.mkdir(exist_ok=True)
        if(ad/'START.json').exists():raise RuntimeError('Unfinished prior attempt: preserve; explicit recovery required, no silent rerun')
        for p in [port,port+1,port+2,tm]:
            s=socket.socket();occupied=s.connect_ex(('127.0.0.1',p))==0;s.close();assert not occupied,('occupied_slot',p)
        config={**method['config'],'output_dir':str(ad/'trace'),'eval_seed':job['seed'],'method_config_sha':method['config_sha256']};write(ad/'agent.json',config);subprocess.run(['chown','-R','cwcarla:cwcarla',str(out)],check=True)
        cmd,env=build_launch(f,job,method,ad,slot);cp=ad/'checkpoint.json'
        rec=dict(started_utc=now(),freeze_sha256=freeze_sha,job=job,method=method['id'],seed=job['seed'],route=job['route'],slot=slot,rpc_port=port,tm_port=tm,cohort='R18_followup_v02_confirmatory_targeted_new12_three_seeds',attempt=attempt,config_sha256=sha(ad/'agent.json'),template_config_sha256=method['config_sha256'],code_directory=f['code_directory'],source_hashes=f['file_sha256'],external_source_hashes=f['external_file_sha256'],command=cmd,physical_output=str(out),logical_output=str(logical),terminal=False)
        write(ad/'START.json',rec);begin=time.monotonic()
        with(ad/'runner.log').open('w')as log:
            p=subprocess.Popen(cmd,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(ad/'RUNNER_PID.json',dict(pid=p.pid,birth_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]))
            timed_out=False
            try:rc=p.wait(timeout=f['failures']['episode_wall_seconds'])
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                rc=p.returncode
        clean=server_cleanup(ad);records=[];parse_errors=[]
        if cp.exists():
            try:records=read(cp).get('_checkpoint',{}).get('records',[])
            except Exception as e:parse_errors.append(repr(e))
        if len(records)>1:parse_errors.append('unexpected multiple official route records in single-route attempt')
        traces=[];tp=ad/'trace/trace.jsonl'
        if tp.exists():
            for line in tp.read_text().splitlines():
                try:traces.append(json.loads(line))
                except ValueError:parse_errors.append('truncated trace row preserved')
        steps=decision_steps(traces,method['kind']);started=steps>0;record=records[-1]if records else None;terminal=terminal_attempt(record,steps,attempt);prefix=None;interface=None
        if started:
            try:
                if method['kind']=='internal':
                    sys.path.insert(0,str(RUN/f['code_directory']));from validate_actual_episode_prefix import validate
                    prefix=validate(ad/'trace');write(ad/'ACTUAL_PREFIX_VALIDATION.json',prefix)
                    interface=internal_identity_validation(ad/'trace',method);interface['actual_prefix_passed']=prefix['passed'];interface['passed']=interface['passed']and prefix['passed']
                else:
                    mp=ad/'trace/agent_metadata.json';metadata=read(mp)if mp.exists()else None;interface=native_trace_validation(traces,metadata)
                    identity=native_identity_validation(metadata,method);interface['native_identity']=identity;interface['passed']=interface['passed']and identity['passed']
                write(ad/'INTERFACE_VALIDATION.json',interface)
            except Exception:interface=dict(passed=False,error=traceback.format_exc());write(ad/'INTERFACE_VALIDATION.json',interface)
        integrity_errors=[]
        try:_,end_sha=frozen();assert end_sha==freeze_sha
        except Exception:integrity_errors.append(traceback.format_exc())
        rec.update(ended_utc=now(),wall_seconds=time.monotonic()-begin,returncode=rc,wall_timeout=timed_out,control_steps_logged=len(traces),model_control_steps=steps,decision_started=started,official_record=record,terminal=terminal,score_if_missing=0,missing_record_events='UNKNOWN'if record is None else None,parse_errors=parse_errors,source_integrity_errors=integrity_errors,owned_server_cleanup=clean,prefix_validation=None if prefix is None else{k:v for k,v in prefix.items()if k!='rows'},interface_validation=interface)
        write(ad/'ATTEMPT.json',rec)
        with(Q/'attempts.jsonl').open('a')as log:fcntl.flock(log,fcntl.LOCK_EX);log.write(json.dumps(rec)+'\n');log.flush()
        if terminal:
            write(out/'FINAL.json',rec);gate=bool(started and interface and interface['passed']and not parse_errors and not integrity_errors)
            write(Q/'jobs'/f'{index:03d}.json',dict(utc=now(),job=job,final=str(out/'FINAL.json'),logical_final=str(logical/'FINAL.json'),gate_passed=gate,decision_started=started,official_record_present=record is not None,prefix_validation=rec['prefix_validation'],interface_validation=interface,freeze_sha256=freeze_sha))
            if not gate:write(Q/'INTERFACE_STOP.json',dict(utc=now(),job=job,reason='Input/control/interface validity failed or no decision started; no downgrade; active jobs finish, new dispatch stops',final=str(out/'FINAL.json')))
            return
        time.sleep(2)
def coordinator():
    # Validate before creating any queue/lock. Missing final freeze has no side effects.
    f,freeze_sha=frozen();guard=lock(Q/'coordinator.lock');globalguard=lock(RUN/'runtime/SCIENTIFIC_EXECUTION.lock');running={};logs={};done={}
    for i in range(216):
        p=Q/'jobs'/f'{i:03d}.json'
        if p.exists():done[i]=read(p);assert done[i]['freeze_sha256']==freeze_sha
    assert all(x['gate_passed']for x in done.values()),'Unresolved failed gate cannot be bypassed by removing STOP'
    validate_pending_pairs=True
    while len(done)<216 or running:
        changed=validate_pending_pairs;validate_pending_pairs=False
        for slot,(index,p)in list(running.items()):
            if p.poll()is not None:
                logs.pop(slot).close();running.pop(slot);jp=Q/'jobs'/f'{index:03d}.json'
                if p.returncode or not jp.exists():write(Q/'INTERFACE_STOP.json',dict(utc=now(),reason='worker ended before accounted FINAL',job_index=index,returncode=p.returncode));done[index]=dict(gate_passed=False,worker_error=True)
                else:done[index]=read(jp)
                changed=True
        if changed and not(Q/'INTERFACE_STOP.json').exists():
            try:
                from confirmation_input_guard import check_ready_pairs
                reports=check_ready_pairs(f,Q,done,RUN)
                if any(not r['passed']for r in reports):write(Q/'INTERFACE_STOP.json',dict(utc=now(),reason='Completed primary-pair input/pool validity gate failed; scores not inspected',failed_pairs=[dict(route=r['route'],seed=r['seed'],arms=r['arms'])for r in reports if not r['passed']]))
            except Exception:write(Q/'INTERFACE_STOP.json',dict(utc=now(),reason='Input guard execution error requires diagnosis',error=traceback.format_exc()))
        stop=(Q/'INTERFACE_STOP.json').exists()or(Q/'PAUSE_NEW_DISPATCH').exists()
        if not stop:
            for slot in range(3):
                if slot in running:continue
                i=next_job(f['schedule'],done,{i for i,_ in running.values()},slot)
                if i is None:continue
                root=Path(f['storage_roots'][f['schedule'][i]['volume']]);fs=os.statvfs(root)
                if fs.f_bavail*fs.f_frsize<6*1024**3:write(Q/'PAUSE_NEW_DISPATCH',dict(utc=now(),reason='target volume below6GiB; preserve assets',root=str(root)));stop=True;break
                lp=Q/'logs'/f'worker_{i:03d}.log';lp.parent.mkdir(parents=True,exist_ok=True);log=lp.open('a');p=subprocess.Popen([sys.executable,__file__,'worker',str(i),str(slot)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[slot]=(i,p);logs[slot]=log
        write(Q/'QUEUE_STATUS.json',dict(utc=now(),pid=os.getpid(),phase='CONFIRMATION',freeze_sha256=freeze_sha,completed_jobs=len(done),expected_jobs=216,remaining_jobs=216-len(done),active=[dict(slot=s,index=i,pid=p.pid,job=f['schedule'][i])for s,(i,p)in running.items()],dispatch_stopped=stop,confirmation_started=True,additional_budget_spent=12))
        if stop and not running:break
        if len(done)==216 and not running:break
        time.sleep(10)
    final=dict(utc=now(),freeze_sha256=freeze_sha,status='CONFIRMATION_MATRIX_COMPLETE'if len(done)==216 and all(x.get('gate_passed')for x in done.values())else'STOPPED_FOR_INTERFACE_OR_INFRASTRUCTURE',completed_jobs=len(done),expected_jobs=216,confirmation_started=True,requires_full_analysis_and_delivery=True)
    with(Q/'coordinator_history.jsonl').open('a')as log:log.write(json.dumps(final)+'\n')
    write(Q/'COORDINATOR_FINAL.json',final)
if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='worker':worker(int(sys.argv[2]),int(sys.argv[3]))
    elif len(sys.argv)==1:coordinator()
    else:raise SystemExit('Only coordinator or worker INDEX SLOT; final scientific freeze required')
