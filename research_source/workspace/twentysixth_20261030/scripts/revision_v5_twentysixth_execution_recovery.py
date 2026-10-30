"""Recover original controller-local bytes without a native/local round trip.

For appended H3 candidates the old archive saves recovered native states, whose
inverse coordinate transform is not bit-exact at geometric ties. Re-execute the
unchanged submitted candidate in its original 12-wide temporary slot. No world
model or future label is consulted, and archived native states are not changed.
"""
import numpy as np
from revision_v5_twentyfirst_full import history
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution

def recover(base, a, current, row, logs):
    h=history(row,logs);ref=int(a['valid'].argmax());slot=11 if ref!=11 else 10
    o=np.asarray(current['rear_axle_pose']);c,s=np.cos(o[2]),np.sin(o[2]);rot=np.array([[c,-s],[s,c]])
    ex=[base['execution']];full=[base['execution_full_local']];errors=[]
    for j in range(12,len(a['valid'])):
        ca=a['candidates'][:12].copy();valid=a['valid'][:12].copy();ca[slot]=a['candidates'][j];valid[slot]=True
        sampled,native,_=predict_vector_controller_execution(h,ca,valid);n=native[slot]
        errors.append(float(abs(n-a['native'][j]).max()))
        np.testing.assert_allclose(n,a['native'][j],atol=1e-7,rtol=0)
        local=n.copy();local[:,:2]=(n[:,:2]-o[:2])@rot;local[:,2]=np.arctan2(np.sin(n[:,2]-o[2]),np.cos(n[:,2]-o[2]))
        local[:,3:5]=n[:,3:5]@rot;local[:,5:7]=n[:,5:7]@rot
        ex.append(sampled[slot:slot+1]);full.append(local[None])
    out={k:v.copy()for k,v in base.items()};out['candidate_valid']=a['valid'].copy()
    out['execution']=np.concatenate(ex);out['execution_full_local']=np.concatenate(full)
    return out,dict(contract='same submitted candidate, original12 temporary slot, original controller, no world or label access',native_maxerrors=errors)

if __name__=='__main__':
    from revision_v5_twentysixth_common import *
    from revision_v5_twentysixth_representation import representations
    sid='study_scene_001009';src=B24/'validation/exposed408';r=next(r for r in json.load(open(src/'choices.json'))if r['scene_id']==sid)
    a=dict(np.load(src/'policy_inputs/PNC_H3P'/sid/'features.npz'));base=dict(np.load(src/'base'/sid/'world_assets.npz'));cur=json.load(open(src/'base'/sid/'current.json'))
    asset,rec=recover(base,a,cur,r,{'hashes':set()});old=np.load(B25/'validation/exposed408/scenes'/sid/'matched_predictions.npz')
    x,rr,_,_=representations(asset,cur,old['current'],old['world']);print(rec,rr,flush=True)
    write(BASE/'diagnostics/controller_recovery.json',dict(scene_id=sid,execution=rec,representation=rr))
