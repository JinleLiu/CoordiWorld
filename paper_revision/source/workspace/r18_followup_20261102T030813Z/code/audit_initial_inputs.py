"""Saved first-decision input/pool audit, no outcomes, policy changes or simulation."""
from pathlib import Path
import json,hashlib,datetime,itertools,math
import numpy as np
RUN=Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z')
def read(p):return json.loads(p.read_text())
def jl(p):return [json.loads(x)for x in p.open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(frames):
    tracks={};out=[]
    for f in frames:
        g={k:v for k,v in f.items()if k not in ['token','scene_token','log_name','anns']}
        an=f['anns'];g['anns']={k:v for k,v in an.items()if k not in ['track_tokens','instance_tokens']}
        for key in ['track_tokens','instance_tokens']:
            ids=[]
            for token in an[key]:
                if token not in tracks:tracks[token]=len(tracks)
                ids.append(tracks[token])
            g['anns'][key]=ids
        def scrub(v):
            if isinstance(v,str)and ':actor:'in v:
                if v not in tracks:tracks[v]=len(tracks)
                return 'actor:'+str(tracks[v])
            if isinstance(v,dict):return {k:scrub(x)for k,x in v.items()}
            if isinstance(v,list):return [scrub(x)for x in v]
            return v
        out.append(scrub(g))
    return out
def differences(a,b,path='',out=None):
    out=[]if out is None else out
    if isinstance(a,dict)and isinstance(b,dict):
        if set(a)!=set(b):out.append(dict(path=path,kind='keys',a=sorted(a),b=sorted(b)))
        for k in a.keys()&b.keys():differences(a[k],b[k],path+'/'+str(k),out)
    elif isinstance(a,list)and isinstance(b,list):
        if len(a)!=len(b):out.append(dict(path=path,kind='length',a=len(a),b=len(b)))
        for i,(x,y)in enumerate(zip(a,b)):differences(x,y,path+'/'+str(i),out)
    elif isinstance(a,(int,float))and isinstance(b,(int,float))and not isinstance(a,bool)and not isinstance(b,bool):
        if not(math.isfinite(a)and math.isfinite(b)):out.append(dict(path=path,kind='nonfinite'))
        elif a!=b:out.append(dict(path=path,kind='numeric',abs_delta=abs(a-b),a=a,b=b))
    elif a!=b:out.append(dict(path=path,kind='value',a=a,b=b))
    return out
def load_job(j):
    fp=Path(j['final']);f=read(fp);tr=fp.parent/f"attempt{f['attempt']}"/'trace';frames=read(tr/'first_four_frames.json');first=next(iter(jl(tr/'policy/plans.jsonl')));traces=jl(tr/'trace.jsonl')
    ap=tr/'policy'/f"plan_{first['replan']:05d}.npz"
    with np.load(ap,allow_pickle=False)as z:
        arrays={k:np.array(z[k][:12])for k in ['candidates','native','common_physical_mask','x674']}
    firststep=next(r['step']for r in traces if r['policy_replans']>0)
    return dict(final=str(fp),final_sha=sha(fp),frames=frames,normalized=normalized(frames),first_plan=first,arrays=arrays,traces=traces,firststep=firststep,npz_sha=sha(ap),frame_sha=sha(tr/'first_four_frames.json'))
def compare(a,b):
    d=differences(a['normalized'],b['normalized']);categories={}
    for v in d:
        path=v['path'].split('/');key=path[2]if len(path)>2 else v['path'];g=categories.setdefault(key,dict(count=0,max_numeric_abs_delta=0.));g['count']+=1;g['max_numeric_abs_delta']=max(g['max_numeric_abs_delta'],v.get('abs_delta',0))
    pools={}
    for k in a['arrays']:
        x,y=a['arrays'][k],b['arrays'][k];eq=x.shape==y.shape and bool(np.array_equal(x,y));pools[k]=dict(a_shape=list(x.shape),b_shape=list(y.shape),exact_equal=eq,max_abs_delta=float(np.max(abs(x.astype(float)-y.astype(float))))if x.shape==y.shape and x.size else None)
    bt={r['step']:r for r in b['traces']};div=None;pre=[]
    for x in a['traces']:
        if x['step']not in bt:continue
        y=bt[x['step']];cd=max(abs(x['control'][k]-y['control'][k])for k in ['steer','throttle','brake'])
        if div is None and cd>1e-5:div=dict(step=x['step'],a_sim_seconds=x['sim_seconds'],b_sim_seconds=y['sim_seconds'],max_control_abs_delta=cd)
        if x['step']<min(a['firststep'],b['firststep']):pre.append(dict(control=cd,position=float(np.linalg.norm(np.array(x['body_pose'][:2])-y['body_pose'][:2])),velocity=float(np.linalg.norm(np.array(x['velocity_world'])-y['velocity_world']))))
    actor_sets=[]
    for af,bf in zip(a['frames'],b['frames']):
        def actors(f):
            an=f['anns'];return sorted(json.dumps([n,box,v],sort_keys=True)for n,box,v in zip(an['gt_names'],an['gt_boxes'],an['gt_velocity_3d']))
        aa,bb=actors(af),actors(bf);actor_sets.append(dict(a_count=len(aa),b_count=len(bb),physical_actor_multiset_exact_equal=aa==bb))
    same=not d;bad=bool(same and not all(pools[k]['exact_equal']for k in ['candidates','native','common_physical_mask']))
    return dict(normalized_observations_exact_equal=same,input_difference_count=len(d),input_difference_categories=categories,input_difference_examples=d[:25],actor_sets=actor_sets,original12_pool=pools,unexpected_common_pool_difference_under_identical_input=bad,first_control_divergence=div,first_policy_steps=[a['firststep'],b['firststep']],pre_policy_samples_compared=len(pre),pre_policy_max_control_delta=max((r['control']for r in pre),default=None),pre_policy_max_position_delta_m=max((r['position']for r in pre),default=None),pre_policy_max_velocity_delta_mps=max((r['velocity']for r in pre),default=None),sources=[dict(final=x['final'],final_sha256=x['final_sha'],first4_frames_sha256=x['frame_sha'],first_npz_sha256=x['npz_sha'])for x in [a,b]])
def main():
    active=read(RUN/'runtime/DEVELOPMENT_ACTIVE.json');q=RUN/active['queue_directory'];jobs=[read(p)for p in sorted((q/'jobs').glob('*.json'))];groups={};errors=[];pairs=[]
    for j in jobs:groups.setdefault((j['job']['route'],j['job']['seed']),{})[j['job']['method']]=j
    for (route,seed),arms in sorted(groups.items()):
        cache={}
        for aa,bb in [('Map-CW','Generic-CW'),('Map-CW','Map-Simple'),('Base-CW','Map-CW')]:
            if aa not in arms or bb not in arms:continue
            try:
                for name in [aa,bb]:
                    if name not in cache:cache[name]=load_job(arms[name])
                pairs.append(dict(route=route,seed=seed,arms=[aa,bb],**compare(cache[aa],cache[bb])))
            except Exception as e:errors.append(dict(route=route,arms=[aa,bb],error=repr(e)))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='SAVED_DEVELOPMENT_INPUT_DIAGNOSTIC',freeze_sha256=sha(RUN/active['freeze']),accounted_jobs=len(jobs),pairs=pairs,errors=errors,identical_input_pairs=sum(r['normalized_observations_exact_equal']for r in pairs),common_pool_failures=sum(r['unexpected_common_pool_difference_under_identical_input']for r in pairs),scope='Current observed ego/actors/map lights/controller state; only episode IDs and consistently remapped actor identities excluded. Absolute timestamps, actor order, all numeric values retained. Exact-input comparisons are stricter than physical tolerances; differences need cause review, not automatic route exclusion. Post-control divergence is own-history closed loop.',no_new_episodes=True,no_score_read=True,source_sha256=sha(Path(__file__)))
    out=RUN/'diagnostics/initial_input_audit';out.mkdir(parents=True,exist_ok=True);p=out/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');p.write_text(json.dumps(result,indent=2)+'\n');(out/'LATEST.json').write_text(json.dumps(result,indent=2)+'\n');print('TAKEOVER_JSON_BEGIN');print(json.dumps(result));print('TAKEOVER_JSON_END')
if __name__=='__main__':main()
