"""Matched generator ablation: current plan profile, same own-history EC.
The padded buffer below is only an implementation device, explicitly NOT an
issued historical plan. It contains current-only predicted execution.
"""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_factory import continuation,ec_matrix
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
from revision_v5_seventeenth_live import b
import numpy as np,time,argparse,copy
def current_profile_candidate(native,choice,current_time,previous_time):
 off=round((current_time-previous_time)/1e5);profile=native[choice];synthetic_buffer=np.concatenate([np.repeat(profile[:1],off,axis=0),profile]);candidate,detail=continuation(native[choice,0],synthetic_buffer,current_time,previous_time,2.,'dynamic');detail.update(profile_source='current base selected candidate execution; NOT historical issued plan',padding_steps=off,history_used_for_profile=False);return candidate,detail
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='oof');args=ap.parse_args();phase=args.phase;src=BASE/'probe80'/phase;out=BASE/'null80'/phase;out.mkdir(parents=True,exist_ok=True);reg=json.load(open(src/'result.json'));last=None;rows=[];start=time.monotonic();record('current_profile_null_'+phase,'Need distinguish ownhistoricalprofile benefit from simply generating another smooth candidate','Same dynamic2 integrator, oneproposal, current selected nativeprofile replacesownpriorprofile; actual ownpriorEC remains','same80TRAINPNC ownOOFhistories; same trigger/admission/readoutbudget')
 for r in reg['rows']:
  sid=r['scene_id'];d=out/sid;d.mkdir(exist_ok=True);a=np.load(src/sid/'inputs.npz');cand,detail=current_profile_candidate(a['old_native'],r['old_choice'],r['timestamp_us'],r['previous_timestamp_us']);s=r['source']
  if last!=s['log_name']:frames=b.load_log(ROOT/'data/openscene/trainval'/(s['log_name']+'.pkl'));last=s['log_name']
  h=frames[s['index']-3:s['index']+1];_,full,_=predict_vector_controller_execution(h,cand[None],np.ones(1,bool));ex=full[0][None];ec=ec_matrix(ex,a['previous_native'],r['offset']);np.savez_compressed(d/'inputs.npz',new_candidates=cand[None],new_native=ex,new_EC=ec);rr=copy.deepcopy(r);rr.update(variants=['current_profile'],new_EC=ec.tolist(),details={'current_profile':detail});write(d/'record.json',rr);rows.append(rr)
 write(out/'result.json',dict(complete=True,created_utc=now(),rows=rows,seconds=time.monotonic()-start,counts=dict(on_trigger=sum(r['trigger']and r['new_EC'][0]>0 for r in rows))));print('B23_NULL',json.load(open(out/'result.json'))['counts'],flush=True)
if __name__=='__main__':main()
