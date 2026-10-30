"""R8 full executed-horizon current-map support; no future objects or labels."""
import argparse,datetime as dt
from revision_v5_eighteenth_common import *
NAMES=['map_known_full_fraction','map_known_all','observed_offroad_any','observed_offroad_fraction','last_0p9_offroad_fraction','first_offroad_time_over4','last_offroad_time_over4','multiple_lane_full_fraction']
def features(area,valid):
    k=area['known'];bad=area['nondrive']&k;anybad=bad.any(1);first=np.where(anybad,bad.argmax(1)/40,1.);last=np.where(anybad,1-bad[:,::-1].argmax(1)/40,0.);f=np.stack([k.mean(1),k.all(1),anybad,bad.mean(1),bad[:,32:].mean(1),first,last,(area['multiple']&k).mean(1)],-1).astype(np.float32);return np.where(valid[:,None],f,0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',default='train',choices=['train','dev']);args=p.parse_args();out=BASE/'features';out.mkdir(exist_ok=True);data=np.load(B16/'features'/('S64_'+args.phase+'.npz'));ids=data['scene_ids'];valid=np.load(B16/'train_inputs.npz')['valid']if args.phase=='train'else np.array([np.load(B16/'features/assets_dev'/str(s)/'assets.npz')['candidate_valid']for s in ids]);ff=[]
    for i,sid in enumerate(ids):
        a=np.load(B16/'features'/('S64_'+args.phase)/(str(sid)+'.npz'))
        ff.append(features({k:a['area_'+k]for k in ['known','nondrive','multiple']},valid[i]))
    np.savez_compressed(out/('R8_'+args.phase+'.npz'),features=np.stack(ff),scene_ids=ids,names=np.array(NAMES));write(out/('R8_'+args.phase+'_receipt.json'),dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha=sha(__file__),input_source='B16 saved S64 area trace, all41 execution steps; S64 summary used onlyfirst32 for TTCprojection',dimensions=8,labels_used=False,source_domain='current map with supportmask; human penalty filtering unknown; not an infallible DAC certificate'))
if __name__=='__main__':main()
