"""Serial fresh-input batch1 timing of the frozen R14 primary, after scoring.
Eight development pairs chosen by observable trigger, never outcome value.
"""
from revision_v5_twentyfourth_common import *
from revision_v5_twentythird_timing_active import infer
from revision_v5_twentythird_features import Pipeline
from revision_v5_twentythird_policy import PNC
from revision_v5_twentythird_memo import install
from revision_v5_twentyfirst_full import history
import numpy as np,torch,yaml,time,platform

def main():
    out=BASE/'timing';out.mkdir(exist_ok=True);examples=[]
    for scope in ['full140','exposed408']:
        folder=BASE/'validation'/scope;by={r['scene_id']:r for r in json.load(open(folder/'choices.json'))}
        records=[r for r in json.load(open(folder/'history_contracts.json'))if r['method']=='PNC_H3P']
        for active in [False,True]:
            for r in sorted([r for r in records if r['trigger']==active],key=lambda x:x['current'])[:2]:
                examples.append(dict(scope=scope,contract=r,previous=by[r['previous']],current=by[r['current']],stratum='active'if active else'inactive'))
    assert len(examples)==8
    write(out/'PREDECLARED_SAMPLE.json',dict(created_utc=now(),examples=examples,selection='first2 by scene ID in each observable trigger stratum and each development cohort; no outcome filter'))
    logs={'hashes':set()}
    for r in examples:
        r['ph']=history(r['previous'],logs);r['h']=history(r['current'],logs)
    t=time.perf_counter();pipe=install(Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml'))));model=PNC();load_seconds=time.perf_counter()-t
    freeze=json.load(open(BASE/'DEVELOPMENT_FREEZE.json'))
    assert model.bindings==freeze['models']['models_PNC']
    records=[]
    for repeat in range(3):
        for example in examples:
            r=example['current'];p=example['previous'];ph=example['ph'];h=example['h'];scope=example['scope'];folder=BASE/'validation'/scope
            prior,pa,pr=infer(pipe,model,ph,p['source']['sha256']);native,a,rec=infer(pipe,model,h,r['source']['sha256'],prior,int(ph[-1]['timestamp']))
            wanted=np.load(folder/'scenes'/r['scene_id']/'inputs.npz')['candidates'][r['choices']['PNC_H3P']]
            prior_wanted=np.load(folder/'base'/p['scene_id']/'native.npz')['states'][p['choices']['PNC_H3P']]
            np.testing.assert_allclose(prior,prior_wanted,atol=1e-7,rtol=0);assert np.array_equal(a['candidates'][rec['choice']],wanted),(r['scene_id'],'timing path differs from frozen choice')
            rec.update(scope=scope,scene_id=r['scene_id'],repeat=repeat,stratum=example['stratum'],submission_sha=array_sha(wanted),frozen_submission_exact=True);records.append(rec)
            print('R14_TIMING',repeat,r['scene_id'],example['stratum'],round(rec['total_ms'],3),flush=True)
    summary={}
    for name in ['all','active','inactive']:
        values=np.array([r['total_ms']for r in records if r['repeat']>0 and(name=='all'or r['stratum']==name)])
        summary[name]=dict(n=len(values),p50_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),mean_ms=float(values.mean()),max_ms=float(values.max()))
    write(out/'result.json',dict(complete=True,created_utc=now(),records=records,summary=summary,model_load_seconds=load_seconds,models=model.bindings,source_sha=sha(__file__),helper_sha=sha(ROOT/'twentythird_20261029/scripts/revision_v5_twentythird_timing_active.py'),
        hardware=dict(GPU=torch.cuda.get_device_name(),CPU=platform.processor(),torch_threads=torch.get_num_threads()),
        scope='serial batch1 fresh current four-frame input to frozen PNC_H3P submission; includes proposer/map/world/controller/features/six heads/triggered H3 supply/P0. Excludes input file I/O, model initialization, image perception, communication and offline native scoring. First repeat excluded.',
        limitation='8 deliberately stratified development pairs, 16 warm calls; not population p95 or a real-time guarantee; no A100 timing conversion; previous-frame compute measured outside current call'))
    print('R14_TIMING_SUMMARY',summary,flush=True)
if __name__=='__main__':main()
