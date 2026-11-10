"""Offline safety/accounting tests. Never dispatches an episode or imports a model."""
from pathlib import Path
import sys,json,hashlib,copy,tempfile,ast
from collections import Counter
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN/'versions/confirmation_v01'))
import confirmation_contract as c
import confirmation_queue as q
def read(p):return json.loads((RUN/p).read_text())
def rejects(fn):
    try:fn()
    except (AssertionError,FileNotFoundError,RuntimeError):return
    raise AssertionError('expected refusal')
def test():
    checks=[];parent=read('protocol/DEVELOPMENT_REPAIR_FREEZE_v02.json');matched=read('protocol/MATCHED_DEVELOPMENT_FREEZE_v01.json');assets=read('protocol/CONFIRMATION_ASSET_PROBE.json');routes=read('protocol/CONFIRMATION_ROUTE_PLAN_v01.json')['ordered_routes']
    roots=dict(data='/original/private/confirmation_artifacts',root='/second/private/confirmation_artifacts')
    for rstar in ['CW','Simple']:
        methods={}
        for label in c.selected_labels(rstar):
            if label in c.INTERNAL_LABELS:
                m=copy.deepcopy({**parent['methods'],**matched['methods']}[label]);m.update(kind='internal',agent=str(q.RUN/parent['code_directory']/'cw_followup_agent.py'))
                # Development uses compact JSON; matching/native use default
                # separators. Retain the existing digest with explicit encoding.
                m['config_hash_encoding']='default_sorted_json_utf8'if label in matched['methods']else'compact_sorted_json_utf8'
            else:
                m=copy.deepcopy(assets['methods']['World4Scorer_route_tp'if label=='W4-native'else'TFpp_single']);m.update(kind='native_public',config_sha256=m['config_sha'],config_hash_encoding='default_sorted_json_utf8')
                m['expected_metadata']=dict(method='World4Scorer_route_tp',official_commit=m['native_commit'],checkpoint_sha256=m['checkpoint_sha'])if label=='W4-native'else dict(method='TFpp_all_towns_seed0',official_commit='48e6a6dbcb133220333bd65e242fbc2b492bb15b',native_model_count=1)
            methods[label]=m
        f=dict(phase='CONFIRMATION',confirmation_authorized_by_this_file=True,T0_utc='2026-11-02T03:08:13+00:00',episode_budget=216,routes=routes,seeds=[17,23,42],methods=methods,r_star=rstar,primary_directions=['Map-r* minus Generic-r*','Map-CW minus Map-Simple'],interface='L1',new_training=False,previously_exposed_routes=['3514','2513','27506','26956','2403','14909'],storage_roots=roots,failures=dict(predecision_infrastructure_retries=1,after_decision_retries=0,missing_official_score=0,missing_official_events='UNKNOWN',episode_wall_seconds=3600,RPC_timeout_seconds=600),slots=parent['slots'],code_directory=parent['code_directory'],route_xml='protocol/PENDING_NOT_FROZEN.xml')
        f['schedule']=c.make_schedule(routes,f['seeds'],rstar,roots);assert c.validate_matrix(f)
        assert all(Counter(j['slot']for j in f['schedule']if j['method']==label)=={0:12,1:12,2:12}for label in methods)
        done=set()
        for _ in range(72):
            for slot in range(3):
                i=c.next_job(f['schedule'],done,set(),slot);assert i is not None and i not in done;done.add(i)
        assert len(done)==216 and all(c.next_job(f['schedule'],done,set(),s)is None for s in range(3))
        for job in f['schedule']:
            m=methods[job['method']];cmd,env=q.build_launch(f,job,m,Path('/not/executed/attempt1'),job['slot']);assert cmd[cmd.index('--track')+1]==m['track']and cmd[cmd.index('--agent')+1]==m['agent']
            assert cmd[4]==m['python']and cmd[cmd.index('--traffic-manager-seed')+1]==str(job['seed'])
            assert c.physical_output(job,m,roots).parent==Path(job['physical_block'])
            if m['id']=='TFpp_single':assert env['STOP_CONTROL']=='1'and env['DEBUG_CHALLENGE']=='0'
            if m['id']=='World4Scorer_route_tp':assert all(p in env['PYTHONPATH'].split(':')for p in m['pythonpath'])
        for mutation in ['duplicate_job','reversed_primary','changed_readout','exposed_route','over_budget','native_as_internal']:
            bad=copy.deepcopy(f)
            if mutation=='duplicate_job':bad['schedule'][-1]=bad['schedule'][0]
            elif mutation=='reversed_primary':bad['primary_directions'][1]='Map-Simple minus Map-CW'
            elif mutation=='changed_readout':bad['methods']['Map-'+rstar]['config']['readout']='changed'
            elif mutation=='exposed_route':bad['previously_exposed_routes'].append(routes[0]['id'])
            elif mutation=='over_budget':bad['episode_budget']=217
            elif mutation=='native_as_internal':bad['methods']['W4-native']['kind']='internal'
            rejects(lambda:c.validate_matrix(bad))
        checks.append(rstar+':216 unique balanced jobs, native/internal commands, six corruption refusals')
    assert not c.terminal_attempt(None,0,1)and c.terminal_attempt(None,0,2)and c.terminal_attempt(None,1,1)
    assert c.terminal_attempt({'status':'Completed'},0,1)and c.terminal_attempt({'status':'Failed-TickRuntime'},1,1)
    assert c.decision_steps([{'policy_replans':0},{'policy_replans':1}],'internal')==1
    assert c.decision_steps([{'initialized':False}],'native_public')==1
    checks.append('bounded predecision retry and terminal postdecision failure; no best retry')
    good=dict(sim_seconds=0.1,pose=[0,0,0],velocity_world=[0,0,0],control=dict(steer=0,throttle=0,brake=1))
    assert c.native_trace_validation([good],{'method':'native'})['passed']
    assert not c.native_trace_validation([good,good],{'method':'native'})['passed']
    assert not c.native_trace_validation([dict(good,control=dict(steer=float('nan'),throttle=0,brake=1))],{'method':'native'})['passed']
    assert not c.native_trace_validation([good],None)['passed'];checks.append('native missing metadata, invalid control and nonmonotone timestamps refused')
    for label in ['W4-native','TFpp-single']:
        m=methods[label];metadata={**m['expected_metadata'],'config':{**m['config'],'output_dir':'/not/executed','eval_seed':17,'method_config_sha':m['config_sha256']}}
        assert c.native_identity_validation(metadata,m)['passed'];metadata['config']['method_config_sha']='wrong'
        assert not c.native_identity_validation(metadata,m)['passed']
    checks.append('native registered method alias, checkpoint identity and runtime config checked')
    with tempfile.TemporaryDirectory()as tmp:
        original=(q.RUN,q.Q);q.RUN=Path(tmp)/'missing_run';q.Q=q.RUN/'confirmation_v01'
        rejects(q.coordinator);assert not q.RUN.exists();q.RUN,q.Q=original
    checks.append('no final freeze: refuses coordinator without creating directories or processes')
    import types
    from confirmation_input_guard import check_ready_pairs
    old_module=sys.modules.get('audit_initial_inputs');mock=types.ModuleType('audit_initial_inputs');mock.load_job=lambda j:dict(normalized={'finite':1})
    mock.compare=lambda a,b:dict(normalized_observations_exact_equal=True,original12_pool={k:dict(exact_equal=True)for k in ['candidates','native','common_physical_mask','x674']})
    mock.differences=lambda a,b:[];sys.modules['audit_initial_inputs']=mock
    primary={j['index']:dict(gate_passed=True)for j in f['schedule']if j['route']==routes[0]['id']and j['seed']==17 and j['method']in ['Map-Simple','Generic-Simple']}
    try:
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);result=check_ready_pairs(f,base/'good',primary,base);assert len(result)==1 and result[0]['passed'];assert check_ready_pairs(f,base/'good',primary,base)==[]
            mock.compare=lambda a,b:dict(normalized_observations_exact_equal=True,original12_pool={'x674':dict(exact_equal=False)})
            assert not check_ready_pairs(f,base/'bad',primary,base)[0]['passed'];rejects(lambda:check_ready_pairs(f,base/'bad',primary,base))
            mock.compare=lambda a,b:dict(normalized_observations_exact_equal=False,original12_pool={'x674':dict(exact_equal=False)})
            result=check_ready_pairs(f,base/'different',primary,base)[0];assert result['passed']and result['input_difference_requires_cause_review']
            mock.differences=lambda a,b:[dict(kind='nonfinite')];assert not check_ready_pairs(f,base/'nan',primary,base)[0]['passed']
    finally:
        if old_module is None:sys.modules.pop('audit_initial_inputs',None)
        else:sys.modules['audit_initial_inputs']=old_module
    checks.append('completed-pair guard detects inconsistent features/nonfinite inputs, preserves input differences, refuses bypass of existing failure')
    def funcs(p):
        tree=ast.parse(p.read_text());return{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
    a=funcs(RUN/'versions/matched_development_v01/matched_queue.py');b=funcs(RUN/'versions/confirmation_v01/confirmation_queue.py')
    for name in ['now','write','lock','_server_cleanup','server_cleanup']:assert a[name]==b[name],name
    checks.append('owned server cleanup and atomic receipt/lock helpers AST-identical to running matched worker')
    return dict(passed=True,scientific_episodes_started=0,confirmation_frozen=False,checks=checks,source_hashes={p.name:c.sha(p)for p in [Path(__file__),RUN/'versions/confirmation_v01/confirmation_contract.py',RUN/'versions/confirmation_v01/confirmation_queue.py',RUN/'versions/confirmation_v01/confirmation_input_guard.py']})
if __name__=='__main__':print(json.dumps(test(),ensure_ascii=False,indent=2))
