"""Sole frozen P: causal generation, then official evaluation; never tune on outcomes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,fcntl,json,pickle,signal,subprocess,sys,time,traceback
from pathlib import Path
import align_train_benchmark as b
import numpy as np
from paired_policy_runtime import make_bank,select

RUN=b.RUN
METHOD='CoordiWorld-AnchoredP-v03'


def lock(name):
    f=(RUN/'locks'/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f


def bindings():
    freeze=RUN/'configs/P_OWN_HISTORY_FREEZE_v01.json';cfg=json.loads(freeze.read_text())
    for rel,h in cfg['sources'].items():assert b.sha(RUN/rel)==h,rel
    assert b.sha(RUN/'configs/P_OWN_HISTORY_INPUT_MANIFEST_v01.json')==cfg['manifest_sha256']
    assert b.sha(RUN/'models/final_P/P.ubj')==cfg['P_model_sha256']
    assert cfg['strategy']=='anchored_P' and cfg['margin']==.005 and cfg['formal_scenes']==1336
    manifest=json.loads((RUN/'configs/P_OWN_HISTORY_INPUT_MANIFEST_v01.json').read_text())
    assert not manifest['capacity_blocked'] and not manifest['missing_metric_cache']
    return cfg,manifest,b.sha(freeze)


def worker(which):
    held=lock(f'P_OWN_WORKER_{which}.lock');cfg,manifest,freeze_sha=bindings()
    audit=RUN/'data/P_own_runtime'/f'{which}_{os.getpid()}';audit.mkdir(parents=True,exist_ok=False)
    from pnc_missing_history_slots_v02 import install
    install(b.pnc.m,audit)
    from current_controller_contract import install as controller_install
    controller=controller_install(b.pnc.ns,audit)
    pipe=b.pnc.m.Pipeline()
    from bounded_graph_cache import install as graph_install
    graph=graph_install(pipe,limit=2)
    assert pipe.checkpoint_sha256==manifest['world_sha256']
    original=b.pnc.FrozenPNC();assert original.bindings==manifest['production_M0_head_bindings']
    model=b.xgb.Booster();model.load_model(RUN/'models/final_P/P.ubj');model.set_param({'nthread':1});assert model.num_features()==2292
    infer,infer_binding=b.diagnostic_infer()
    b.write(audit/'bindings.json',dict(at=b.now(),freeze_sha256=freeze_sha,world_sha256=pipe.checkpoint_sha256,
        M0=original.bindings,P_model_sha256=cfg['P_model_sha256'],infer=infer_binding,controller=controller,graph_cache=graph))
    selected_blocks=set(manifest['worker_blocks'][which]);blocks=[x for x in manifest['blocks'] if x['block'] in selected_blocks]
    fallbacks={x['token']:x for x in manifest['inherited_fallbacks']}
    qpath=RUN/f'queue/P_own_worker_{which}.json';done=generated=reused=changes=0;start=time.monotonic();token=None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen1800s per-input resource watchdog; no automatic retry')))
    try:
        for block in blocks:
            assert b.sha(block['metadata_path'])==block['metadata_sha256']
            previous=None;previous_ts=None;previous_token=None;loaded={}
            for row in block['records']:
                token=row['token'];contract=row['input_contract']
                if contract['causal_reset_before']:previous=None;previous_ts=None;previous_token=None
                assert contract['causal_blocker'] is None
                expected=dict(previous_token=previous_token,previous_timestamp_us=previous_ts,
                    previous_plan_sha256=None if previous is None else b.array_sha(previous),block=block['block'])
                dest=RUN/'predictions'/METHOD/token
                b.write(qpath,dict(at=b.now(),status='RUNNING',pid=os.getpid(),worker=which,current=token,
                    done=done,total=manifest['worker_counts'][which],generated=generated,reused=reused,changed=changes,seconds=time.monotonic()-start))
                if (dest/'receipt.json').exists():
                    rec=json.loads((dest/'receipt.json').read_text());assert rec['status']=='PASS' and rec['freeze_sha256']==freeze_sha and rec['history']==expected
                    assert b.sha(dest/'trajectory.npy')==rec['trajectory_sha256'] and b.sha(dest/'own_predicted_execution.npy')==rec['execution_sha256']
                    previous=np.load(dest/'own_predicted_execution.npy',allow_pickle=False);previous_ts=rec['currenttimestamp'];previous_token=token
                    done+=1;reused+=1;changes+=rec['changed'];continue
                if dest.exists():
                    failed=RUN/'data/failed_P_own_inference'/token/f'{os.getpid()}_{time.time_ns()}';failed.parent.mkdir(parents=True,exist_ok=True);dest.rename(failed)
                dest.mkdir(parents=True);signal.alarm(1800);t=time.monotonic()
                path=row['raw_log']
                if path not in loaded:
                    assert b.sha(path)==row['raw_log_sha256'];loaded={path:pickle.loads(Path(path).read_bytes())}
                h=loaded[path][row['index']-3:row['index']+1]
                assert len(h)==4 and str(h[-1]['token'])==token and int(h[-1]['timestamp'])==contract['current_timestamp_us']
                if previous_ts is not None:assert 0<int(h[-1]['timestamp'])-previous_ts<550000
                if contract['inherited_fallback']:
                    source=fallbacks[token]
                    assert not row['scored'] and b.sha(source['trajectory_path'])==source['trajectory_sha256'] and b.sha(source['execution_path'])==source['execution_sha256']
                    poses=np.load(source['trajectory_path'],allow_pickle=False);issued=np.load(source['execution_path'],allow_pickle=False)
                    selected=baseline=0;detail=dict(inherited_fallback=True,readout_available=False,source=source)
                else:
                    _,a,infer_rec,asset,current,ec,executions,brake_slot=infer(pipe,original,h,row['raw_log_sha256'],previous,previous_ts)
                    n=len(a['valid']);origins=['native_PDM']*min(12,n)
                    if brake_slot is not None:origins[int(brake_slot)]='brake_supply'
                    origins+=['H3_'+x['variant'] for x in infer_rec['variant_steps'] if x['admitted']]
                    assert len(origins)==n
                    a['candidate_origin']=np.array(origins);a['candidate_id']=np.array([token+':'+b.array_sha(x) for x in a['candidates']])
                    a['EC']=np.full(n,np.nan) if ec is None else np.asarray(ec)
                    baseline=int(infer_rec['choice']);heads=original.last_prediction
                    bank=make_bank(a,asset,current,heads,baseline);selected,selector=select(model,bank,cfg['margin'])
                    poses=a['candidates'][selected];issued=executions[selected]
                    np.savez_compressed(dest/'candidate_features.npz',**a)
                    np.savez_compressed(dest/'head_outputs.npz',**heads)
                    np.savez_compressed(dest/'P_readout.npz',M56=bank['M56'][0],**selector)
                    np.savez_compressed(dest/'world_assets.npz',**asset)
                    np.save(dest/'all_predicted_executions.npy',executions)
                    b.write(dest/'current.json',current)
                    # Full numerical readout and world evidence stay on original AutoDL.
                    world_hashes={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=b.array_sha(v)) for k,v in asset.items() if isinstance(v,np.ndarray) and not v.dtype.hasobject}
                    detail=dict(inherited_fallback=False,readout_available=True,M0_infer=infer_rec,
                        candidate_bank_sha256=b.sha(dest/'candidate_features.npz'),head_outputs_sha256=b.sha(dest/'head_outputs.npz'),
                        P_readout_sha256=b.sha(dest/'P_readout.npz'),world_array_hashes=world_hashes,
                        world_assets_sha256=b.sha(dest/'world_assets.npz'))
                    del asset,bank
                    b.torch.cuda.empty_cache()
                assert poses.shape==(8,3) and issued.shape==(41,11) and np.isfinite(poses).all() and np.isfinite(issued).all()
                np.save(dest/'trajectory.npy',poses);np.save(dest/'own_predicted_execution.npy',issued)
                original_pose=b.R18/'predictions/CoordiWorld-PNC_H3P'/token/'trajectory.npy'
                changed_original=not np.array_equal(poses,np.load(original_pose,allow_pickle=False))
                rec=dict(status='PASS',at=b.now(),token=token,method=METHOD,freeze_sha256=freeze_sha,history=expected,
                    currenttimestamp=int(h[-1]['timestamp']),scored=bool(row['scored']),choice=int(selected),M0_choice=int(baseline),changed=bool(selected!=baseline),
                    changed_from_original=changed_original,original_trajectory_sha256=b.sha(original_pose),
                    trajectory_sha256=b.sha(dest/'trajectory.npy'),execution_sha256=b.sha(dest/'own_predicted_execution.npy'),
                    seconds=time.monotonic()-t,raw_log_sha256=row['raw_log_sha256'],new_outcome_values_read=False,**detail)
                b.write(dest/'receipt.json',rec);signal.alarm(0)
                previous=issued.copy();previous_ts=int(h[-1]['timestamp']);previous_token=token
                generated+=1;done+=1;changes+=selected!=baseline
        b.write(qpath,dict(at=b.now(),status='COMPLETE',pid=os.getpid(),worker=which,done=done,total=manifest['worker_counts'][which],generated=generated,reused=reused,changed=changes,seconds=time.monotonic()-start))
    except Exception:
        signal.alarm(0);failure=dict(at=b.now(),status='FAILED',pid=os.getpid(),worker=which,token=token,done=done,generated=generated,reused=reused,traceback=traceback.format_exc(),automatic_retry=False)
        b.write(RUN/'failures/P_own_history'/f'{which}_{os.getpid()}_{time.time_ns()}.json',failure);b.write(qpath,failure);raise


def supervise():
    held=lock('P_OWN_HISTORY_RUN.lock');cfg,manifest,freeze_sha=bindings()
    started=time.monotonic();attempt=f'{os.getpid()}_{time.time_ns()}';stages=[]
    for stage,script in [('inference',Path(__file__)),('official_evaluation',RUN/'code/evaluate_p_own_history.py')]:
        if stage=='official_evaluation':
            records=[]
            for block in manifest['blocks']:
                for row in block['records']:
                    p=RUN/'predictions'/METHOD/row['token']/'receipt.json';rec=json.loads(p.read_text())
                    assert rec['status']=='PASS' and rec['freeze_sha256']==freeze_sha
                    assert b.sha(p.parent/'trajectory.npy')==rec['trajectory_sha256']
                    records.append(dict(token=row['token'],scored=bool(row['scored']),receipt_sha256=b.sha(p)))
            assert len(records)==2919 and sum(x['scored'] for x in records)==1336
            dest=RUN/'evidence/P_OWN_SELECTION_FROZEN_v01.json'
            if dest.exists():assert json.loads(dest.read_text())['records']==records
            else:b.write(dest,dict(at=b.now(),status='ALL_P_OUTPUTS_FROZEN_BEFORE_NEW_OUTCOME_READS',freeze_sha256=freeze_sha,records=records))
        children=[]
        for i in range(3):
            log=RUN/'logs'/f'P_own_{stage}_{i}_{attempt}.log'
            with log.open('w') as stream:child=subprocess.Popen([sys.executable,str(script),'--worker',str(i)],cwd=RUN,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            children.append((i,child,str(log)))
        while True:
            status=[dict(worker=i,pid=p.pid,exit_code=p.poll(),log=log) for i,p,log in children]
            bad=any(x['exit_code'] not in [None,0] for x in status)
            b.write(RUN/'QUEUE_STATUS.json',dict(at=b.now(),status='P_OWN_WORKER_FAILURE_INSPECT_NO_AUTO_RETRY' if bad else 'P_OWN_HISTORY_RUNNING',stage=stage,pid=os.getpid(),children=status,seconds=time.monotonic()-started,freeze_sha256=freeze_sha))
            if all(x['exit_code'] is not None for x in status):break
            time.sleep(10)
        stages.append(dict(stage=stage,children=status))
        if bad:
            b.write(RUN/'evidence'/f'P_OWN_FAILURE_{attempt}.json',dict(at=b.now(),stages=stages,automatic_retry=False));return 1
    log=RUN/'logs'/f'P_own_aggregate_{attempt}.log'
    with log.open('w') as stream:rc=subprocess.call([sys.executable,str(RUN/'code/evaluate_p_own_history.py'),'--aggregate'],cwd=RUN,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    result=dict(at=b.now(),status='P_OWN_HISTORY_COMPLETE_AWAITING_FINAL_ANALYSIS' if rc==0 else 'P_OWN_AGGREGATE_FAILED',stages=stages,aggregate_exit_code=rc,log=str(log),seconds=time.monotonic()-started,freeze_sha256=freeze_sha)
    b.write(RUN/'QUEUE_STATUS.json',result);b.write(RUN/'evidence'/f'P_OWN_SUPERVISOR_{attempt}.json',result)
    return rc


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is None:sys.exit(supervise())
    else:worker(args.worker)
