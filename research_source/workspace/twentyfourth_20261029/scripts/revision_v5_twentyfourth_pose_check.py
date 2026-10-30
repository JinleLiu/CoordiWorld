"""TRAIN-only frozen-head drift; no protection/development outcomes."""
from revision_v5_twentyfourth_prepare import *
from revision_v5_twentythird_features import b
from coordiworld.revision_v5.component_known_b13 import geometry_known_score
import argparse

def main(limit):
    out=BASE/'train_pose_check';out.mkdir(exist_ok=True)
    rows=json.load(open(B23/'probe80/oof/result.json'))['rows'][:limit]
    pipe=install(Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml'))))
    models={f:Readouts('oof',f)for f in range(3)};records=[];labels=np.load(ROOT/'twentysecond_20261028/supply55/dataset.npz')
    for r in rows:
        check_time();sid=r['scene_id'];d=out/sid;d.mkdir(exist_ok=True)
        if (d/'receipt.json').exists():records.append(json.load(open(d/'receipt.json')));continue
        src=r['source'];path=ROOT/'data/openscene/trainval'/(src['log_name']+'.pkl');assert sha(path)==src['raw_log_sha256']
        frames=b.load_log(path);j=src['index'];h=frames[j-3:j+1]
        a,asset,native,current,receipt=base_inputs(pipe,h,src['raw_log_sha256'])
        old=dict(np.load(B23/'spatial_hybrid_features/oof'/sid/'features.npz'))
        old={k:old[k][:12].copy()for k in a if k!='geo'}
        f=old['features198'];v=old['valid'];g=int(np.where(v,geometry_known_score(f[None],f[None,:,188:],v[None])[0],-np.inf).argmax());old['geo']=np.array(g)
        for key in ['x674','old_x674']:
            p=old[key][:,658:];p[:,8]=p[:,0]-p[g,0];p[:,9]=p[:,4]-p[g,4];p[~v]=0
        pnew=models[r['fold']].predict(a);pold=models[r['fold']].predict(old)
        equal=np.array_equal(a['candidates'],old['candidates'])and np.array_equal(a['valid'],old['valid'])
        rec=receipt|dict(scene_id=sid,drive=r['drive'],fold=r['fold'],source=src,candidates_exact=equal,
            candidate_max_difference=float(abs(a['candidates']-old['candidates']).max()),
            old_choices={k:int(vv['choice'])for k,vv in pold.items()},new_choices={k:int(vv['choice'])for k,vv in pnew.items()},
            feature_max_difference={k:float(abs(a[k]-old[k]).max())for k in ['features198','x674','V24','R8','C32']},
            old_components={k:labels['components'][r['index'],vv['choice']].tolist()for k,vv in pold.items()},
            new_components_if_exact={k:labels['components'][r['index'],vv['choice']].tolist()for k,vv in pnew.items()}if equal else None)
        np.savez_compressed(d/'features.npz',**a);np.savez_compressed(d/'world_assets.npz',**asset);np.savez_compressed(d/'native.npz',states=native)
        write(d/'current.json',current);write(d/'receipt.json',rec);records.append(rec)
        print('R14_TRAIN_POSE',len(records),sid,rec['max_yaw_difference'],rec['old_choices'],rec['new_choices'],flush=True)
    write(out/'result.json',dict(complete=True,created_utc=now(),scenes=len(records),drives=len({r['drive']for r in records}),records=records,
        exact_candidate_pools=sum(r['candidates_exact']for r in records),changed={m:sum(r['new_choices'][m]!=r['old_choices'][m]for r in records)for m in ['PNC','IC0','Geometry']},
        scope='TRAIN old OOF heads, current-input pose correction only; score reuse only for exact trajectory/valid identities'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=80);main(ap.parse_args().limit)
