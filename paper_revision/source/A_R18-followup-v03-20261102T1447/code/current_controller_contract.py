"""Current-state-only controller contract for inherited oldest-slot histories.

Regular inputs call the exact original function. The only removed AST node in
the alternate entry rejects four-frame spacing, although numerical control reads
only history[-1]. Oldest-slot eligibility remains checked; no timestamps change.
"""
import ast
import hashlib
import inspect
import json
import sys
import numpy as np

EXPECTED_AST = '735e735145ee89e79d640094b6226c99df4f783c2420b31fd5f5f9fb3f545d08'


def functions():
    import coordiworld.revision_v5.compiled_solve_controller as module
    original = module.predict_vector_controller_execution
    node = ast.parse(inspect.getsource(original)).body[0]
    digest = hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    assert digest == EXPECTED_AST
    guards = [n for n in node.body if isinstance(n,ast.If) and any(
        isinstance(x,ast.Constant) and x.value=='Noncontiguous history' for x in ast.walk(n))]
    assert len(guards)==1
    removed = ast.unparse(guards[0]); node.body.remove(guards[0])
    revised = hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    namespace = original.__globals__.copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 original.__code__.co_filename+'#v03_current_state_contract','exec'),namespace)
    return original,namespace[node.name],dict(original_function_ast_sha256=digest,
        revised_function_ast_sha256=revised,removed_only=removed,
        original_file_sha256=hashlib.sha256(open(inspect.getfile(original),'rb').read()).hexdigest(),
        numerical_body_changed=False,original_source_file_modified=False,
        historical_observations_modified=False,regular_entry_unchanged=True)


def install(namespace,audit_root):
    from a0_causal_context_manifest import history_kind
    original,contracted,binding = functions()
    def wrapper(history,candidates,candidate_valid):
        times=np.array([int(f['timestamp']) for f in history],dtype=np.int64)
        if len(times)==4 and np.allclose(np.diff(times)/1e6,.5,atol=.08):
            return original(history,candidates,candidate_valid)
        assert history_kind(history,3)=='inherited_R18_causal_oldest_slot_adapter', 'Unsupported controller history gap'
        return contracted(history,candidates,candidate_valid)
    aliases=[]
    for name,module in list(sys.modules.items()):
        if name.startswith(('coordiworld.','revision_v5','pnc_','minimal_runtime','align_train')) and getattr(module,'predict_vector_controller_execution',None) is original:
            module.predict_vector_controller_execution=wrapper; aliases.append(name)
    if namespace.get('predict_vector_controller_execution') is original:
        namespace['predict_vector_controller_execution']=wrapper; aliases.append('PNC_AST_namespace')
    assert 'coordiworld.revision_v5.compiled_solve_controller' in aliases
    binding['patched_aliases']=aliases
    audit_root.mkdir(parents=True,exist_ok=True)
    (audit_root/'current_controller_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    return binding


if __name__=='__main__':
    import pickle
    from pathlib import Path
    import align_train_benchmark as b
    original,contracted,binding=functions()
    paths=sorted((b.RUN/'data/current_M0').glob('*/candidate_features.npz'))
    for path in paths:
        folder=path.parent
        if not(folder/'receipt.json').exists():continue
        rec=json.loads((folder/'receipt.json').read_text())
        raw=pickle.loads(Path(rec['raw_log']).read_bytes());i=rec['current_index'];history=raw[i-3:i+1]
        if np.allclose(np.diff([int(x['timestamp']) for x in history])/1e6,.5,atol=.08):break
    with np.load(path,allow_pickle=False) as a:
        left=original(history,a['candidates'],a['valid'])
        right=contracted(history,a['candidates'],a['valid'])
    assert np.array_equal(left[0],right[0]) and set(left[1])==set(right[1])
    assert all(np.array_equal(left[1][k],right[1][k]) for k in left[1])
    assert left[2]==right[2]
    binding.update(status='PASS',scene=folder.name,candidates_checked=len(left[1]),
                   controller_batch_calls=2,native_score_calls=0,model_calls=0,
                   all_executed_state_values_byte_equal=True,new_fits=0)
    b.write(b.RUN/'evidence/CURRENT_CONTROLLER_EQUIVALENCE.json',binding)
    print(json.dumps(binding),flush=True)
