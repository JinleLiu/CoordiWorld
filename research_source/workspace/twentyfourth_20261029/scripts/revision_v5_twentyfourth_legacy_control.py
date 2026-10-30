"""Small fresh legacy-source control; excludes cache-age as a pose effect.
Run with runtime_s1/src, deliberately NOT R14/runtime_src.
"""
from revision_v5_twentythird_timing_active import infer,Pipeline,PNC,b
from revision_v5_twentythird_memo import install
from revision_v5_twentyfourth_common import *
from coordiworld.data.openscene import ego_pose
import inspect,numpy as np,yaml

def main():
    assert '/runtime_s1/src/' in inspect.getfile(ego_pose)
    out=BASE/'fresh_legacy_control';out.mkdir(exist_ok=True)
    pipe=install(Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml'))))
    models={k:PNC('oof',k)for k in range(3)};records=[]
    for r in json.load(open(B23/'probe80/oof/result.json'))['rows'][:6]:
        so=r['source'];frames=b.load_log(ROOT/'data/openscene/trainval'/(so['log_name']+'.pkl'));j=so['index'];h=frames[j-3:j+1]
        native,a,rr=infer(pipe,models[r['fold']],h,so['raw_log_sha256'])
        old=np.load(B23/'spatial_hybrid_features/oof'/r['scene_id']/'features.npz')
        d=out/r['scene_id'];d.mkdir(exist_ok=True);np.savez_compressed(d/'features.npz',**a)
        records.append(dict(scene_id=r['scene_id'],candidates_exact=bool(np.array_equal(a['candidates'],old['candidates'][:12])),valid_exact=bool(np.array_equal(a['valid'],old['valid'][:12])),candidate_maxerr=float(abs(a['candidates']-old['candidates'][:12]).max()),features198_maxerr=float(abs(a['features198']-old['features198'][:12]).max()),seconds=rr['total_ms']/1000))
        print('FRESH_LEGACY',records[-1],flush=True)
    write(out/'result.json',dict(complete=True,created_utc=now(),records=records,legacy_pose_source=inspect.getfile(ego_pose),legacy_pose_sha=sha(inspect.getfile(ego_pose)),scope='six TRAIN same source generation control; no protected results'))

if __name__=='__main__':main()
