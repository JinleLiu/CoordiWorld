"""Completed-pair input validity only; never reads scores or chooses a method."""
from pathlib import Path
import sys,json,datetime
def check_ready_pairs(f,queue,done,run):
    sys.path.insert(0,str(run/'code'))
    from audit_initial_inputs import load_job,compare,differences
    ids={(j['route'],j['seed'],j['method']):j['index']for j in f['schedule']};reports=[]
    contrasts=[('Map-'+f['r_star'],'Generic-'+f['r_star']),('Map-CW','Map-Simple')]
    for route in f['routes']:
        for seed in f['seeds']:
            for ci,(aa,bb)in enumerate(contrasts):
                indices=[ids[(route['id'],seed,m)]for m in [aa,bb]]
                if not all(i in done and done[i].get('gate_passed')for i in indices):continue
                out=queue/'input_guards'/f"{route['id']}_{seed}_{ci}.json"
                if out.exists():
                    old=json.loads(out.read_text());assert old['passed'],'Existing failed input gate requires diagnosis; never skip it'
                    continue
                out.parent.mkdir(parents=True,exist_ok=True)
                try:
                    a,b=[load_job(done[i])for i in indices];result=compare(a,b)
                    nonfinite=any(d['kind']=='nonfinite'for x in [a,b]for d in differences(x['normalized'],x['normalized']))
                    inconsistent=bool(result['normalized_observations_exact_equal']and not all(v['exact_equal']for v in result['original12_pool'].values()))
                    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),route=route['id'],seed=seed,arms=[aa,bb],passed=not(nonfinite or inconsistent),nonfinite_input=nonfinite,inconsistent_original_pool_or_features_under_identical_input=inconsistent,input_difference_requires_cause_review=not result['normalized_observations_exact_equal'],audit=result,no_score_fields_used=True)
                except Exception as exc:record=dict(route=route['id'],seed=seed,arms=[aa,bb],passed=False,error=repr(exc))
                out.write_text(json.dumps(record,indent=2)+'\n');reports.append(record)
    return reports
