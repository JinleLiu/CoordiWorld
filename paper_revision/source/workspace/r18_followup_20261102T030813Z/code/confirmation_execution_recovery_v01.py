"""Execution-only recovery: serialize bootstrap until START, then keep three slots.

The original frozen coordinator/worker files remain byte-identical. All science,
slot/route/seed order, interface gates and evaluator arguments remain unchanged.
Job34's pre-START lock failure consumes its sole predecision retry: its recovered
worker therefore has one simulator attempt and cannot request attempt2.
"""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,subprocess,sys,time
RUN=Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z')
FREEZE_SHA='c82e2bfdc750850cda5e6633144c79d8117abe8ad9019b3776c432b49dcb9a40'
RECOVERY_JOB=34

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_runtime():
    assert sha(RUN/'protocol/CONFIRMATION_FREEZE_v01.json')==FREEZE_SHA
    plan=json.loads((RUN/'protocol/EXECUTION_RECOVERY_PLAN_v01.json').read_text())
    assert plan['freeze_sha256']==FREEZE_SHA and plan['entry_sha256']==sha(__file__),'Operational entry changed after review'
    directory=RUN/'versions/confirmation_v01';sys.path.insert(0,str(directory));spec=importlib.util.spec_from_file_location('frozen_confirmation_runtime',directory/'confirmation_queue.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def wait_bootstrap(proc,start,timeout_seconds=60):
    deadline=time.monotonic()+timeout_seconds
    while True:
        if Path(start).is_file():return 'START_VISIBLE_INITIALIZATION_LOCK_RELEASED'
        if proc.poll()is not None:return 'WORKER_EXIT_BEFORE_START'
        if time.monotonic()>=deadline:return 'BOOTSTRAP_WAIT_TIMEOUT_NO_NEW_DISPATCH'
        time.sleep(.02)

def validate_before_launch(out,job_index):
    if list(Path(out).glob('attempt*/START.json')):raise RuntimeError('Existing attempt requires explicit recovery, never overwrite or silently rerun')
    if(Path(out)/'FINAL.json').exists():raise RuntimeError('Already-final job must not be launched')

def terminal_after_spent_retry(record,steps,attempt):
    assert attempt==1,'Bootstrap failure consumed the only predecision retry'
    return True

def spent_retry_worker(index,slot):
    assert index==RECOVERY_JOB
    m=load_runtime();f,h=m.frozen();job=f['schedule'][index];assert job['slot']==slot
    event=json.loads((RUN/'confirmation_v01/operational_recovery_v01/BOOTSTRAP_FAILURE_034.json').read_text());assert event['predecision_retry_consumed']==1 and event['scenario_started']is False
    out=m.physical_output(job,f['methods'][job['method']],f['storage_roots']);validate_before_launch(out,index)
    m.write(RUN/'confirmation_v01/operational_recovery_v01/JOB034_RETRY_BUDGET.json',dict(utc=now(),freeze_sha256=h,bootstrap_failures=1,simulator_attempts_allowed=1,after_decision_retries=0,entry_sha256=sha(__file__),original_terminal_function_source_unchanged=True))
    # This only exhausts the existing retry allowance; it does not pass any gate.
    m.terminal_attempt=terminal_after_spent_retry;m.worker(index,slot)

def coordinator():
    m=load_runtime();f,h=m.frozen();real_popen=subprocess.Popen;original=str(RUN/'versions/confirmation_v01/confirmation_queue.py');Q=RUN/'confirmation_v01';op=Q/'operational_recovery_v01';op.mkdir(exist_ok=True)
    def launch(args,*pargs,**kwargs):
        is_worker=isinstance(args,(list,tuple))and len(args)==5 and str(args[1])==original and args[2]=='worker'
        if not is_worker:return real_popen(args,*pargs,**kwargs)
        if(Q/'INTERFACE_STOP.json').exists()or(Q/'PAUSE_NEW_DISPATCH').exists():raise RuntimeError('Stop is set; no further worker launch')
        index=int(args[3]);slot=int(args[4]);job=f['schedule'][index];out=m.physical_output(job,f['methods'][job['method']],f['storage_roots']);validate_before_launch(out,index)
        command=list(args)
        if index==RECOVERY_JOB:command[1]=str(Path(__file__).resolve());command[2]='spent-retry-worker'
        proc=real_popen(command,*pargs,**kwargs);begin=time.monotonic();result=wait_bootstrap(proc,out/'attempt1/START.json')
        m.write(op/f'launch_{index:03d}.json',dict(utc=now(),job_index=index,slot=slot,pid=proc.pid,original_worker_command=list(args),actual_worker_command=command,bootstrap_wait_seconds=time.monotonic()-begin,result=result,freeze_sha256=h,entry_sha256=sha(__file__),science_arguments_unchanged=True))
        if result!='START_VISIBLE_INITIALIZATION_LOCK_RELEASED':m.write(Q/'INTERFACE_STOP.json',dict(utc=now(),reason=result,job_index=index,pid=proc.pid,active_worker_preserved=True))
        return proc
    # Parent-only interposition. Normal workers exec the unchanged frozen entry.
    # Coordinator also retains its original slot/global locks and pair guards.
    m.subprocess.Popen=launch
    try:m.coordinator()
    finally:m.subprocess.Popen=real_popen

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory()as td:
        root=Path(td);child=root/'child.py';child.write_text('''import fcntl,sys,time
from pathlib import Path
p=Path(sys.argv[1]);h=(p/'shared.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);time.sleep(.15);fcntl.flock(h,fcntl.LOCK_UN);(p/sys.argv[2]).write_text('started');time.sleep(.2)
''')
        first=subprocess.Popen([sys.executable,str(child),str(root),'first']);assert wait_bootstrap(first,root/'first')=='START_VISIBLE_INITIALIZATION_LOCK_RELEASED'
        second=subprocess.Popen([sys.executable,str(child),str(root),'second']);assert wait_bootstrap(second,root/'second')=='START_VISIBLE_INITIALIZATION_LOCK_RELEASED';assert first.wait()==0 and second.wait()==0
        failed=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)']);assert wait_bootstrap(failed,root/'absent')=='WORKER_EXIT_BEFORE_START'
        out=root/'out';out.mkdir();validate_before_launch(out,34);(out/'attempt1').mkdir();(out/'attempt1/START.json').write_text('{}')
        try:validate_before_launch(out,34)
        except RuntimeError:pass
        else:raise AssertionError('Existing attempt was accepted')
        assert terminal_after_spent_retry(None,0,1)
        try:terminal_after_spent_retry(None,0,2)
        except AssertionError:pass
        else:raise AssertionError('Extra retry was accepted')
        pending=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.3)']);assert wait_bootstrap(pending,root/'never',.04)=='BOOTSTRAP_WAIT_TIMEOUT_NO_NEW_DISPATCH';assert pending.poll()is None;pending.wait()
    return dict(passed=True,checks=['shared bootstrap serialization prevents lock contention','workers may overlap after bootstrap','pre-START failure remains failure','existing START refused','spent retry is terminal on first simulator attempt','bootstrap timeout preserves existing process'],new_scientific_episodes=0)

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--self-test':print(json.dumps(self_test()))
    elif len(sys.argv)==4 and sys.argv[1]=='spent-retry-worker':spent_retry_worker(int(sys.argv[2]),int(sys.argv[3]))
    elif len(sys.argv)==1:coordinator()
    else:raise SystemExit('Only coordinator, spent-retry-worker INDEX SLOT, or --self-test')
