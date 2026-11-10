"""Current-only diagnosis of frozen M12 kinematic rejection, no labels."""
import json,copy
from pathlib import Path
import numpy as np,yaml
from revision_v5_twentyeighth_policy import array_history
from revision_v5_twentyeighth_map import install
R=Path('<CW_WORKSPACE>');B=R/'twentyeighth_20261031'
def main():
 d=B/'bench2drive/episodes/seed17/3514/CoordiWorld_Full/attempt1/trace';h=array_history(json.load(open(d/'first_four_frames.json')));install(json.load(open(B/'bridge/maps/Town13.json')))
 from coordiworld.data.pdm_proposer import PDMNominalProposer
 from coordiworld.data.openscene import ego_pose,transform_pose
 from nuplan.common.actor_state.state_representation import TimePoint
 p=PDMNominalProposer(R/'data/maps',yaml.safe_load(open(R/'runtime_s1/configs/group_cache_v2.yaml'))['candidates']);original=p.planner.compute_planner_trajectory;captured=[]
 def capture(x):
  z=original(x);captured.append(z);return z
 p.planner.compute_planner_trajectory=capture;ca,valid,meta=p(h);times=np.arange(81)*.1;states=captured[0].get_state_at_times([TimePoint(h[-1]['timestamp']+int(round(t*1e6)))for t in times]);origin=ego_pose(h[-1]);poses=np.array([transform_pose(np.array([s.rear_axle.x,s.rear_axle.y,s.rear_axle.heading]),origin)for s in states]);ca[0]=poses[5::5][:8];v0=np.linalg.norm(h[-1]['ego_dynamic_state'][:2]);rows=[]
 for c,v in zip(ca,meta['variants']):
  f=np.r_[np.zeros((1,3)),c];speed=np.linalg.norm(np.diff(f[:,:2],axis=0),axis=1)/.5;acc=np.diff(np.r_[v0,speed])/.5;yaw=np.diff(np.unwrap(f[:,2]))/.5;rows.append(dict(name=v['name'],max_acc=float(acc.max()),min_acc=float(acc.min()),max_yaw=float(abs(yaw).max()),max_lat=float((speed*abs(yaw)).max()),first_pose=c[0].tolist(),last_pose=c[-1].tolist()))
 out=B/'bridge/nominal_diagnosis';out.mkdir(exist_ok=True);np.savez_compressed(out/'actual_first_state_proposals.npz',dense_nominal=poses,candidates_before_fallback=ca,valid=valid,initial_pose=origin);r=dict(input_source=str(d/'first_four_frames.json'),label_access=False,manifest=meta,diagnostics=rows);(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print('M12_DIAG',rows,flush=True)
 from coordiworld.revision_v5.compiled_solve_controller import trajectory_array
 from coordiworld.revision_v5.compiled_solve_kernel import CompiledPDMSimulator
 from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
 from navsim.common.dataclasses import Scene,SensorConfig
 from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
 from coordiworld.data.pdm_proposer import perturb_nominal
 sc=NavSimScenario(Scene.from_scene_dict_list(h,Path('.'),4,0,SensorConfig.build_no_sensors()),str(R/'data/maps'),'nuplan-maps-v1.0');sampling=TrajectorySampling(num_poses=80,interval_length=.1);raw=trajectory_array(captured[0],sampling,sc.initial_ego_state.time_point);executed=CompiledPDMSimulator(sampling).simulate_proposals(raw[None],sc.initial_ego_state)[0];local=np.array([transform_pose(x[:3],origin)for x in executed]);np.testing.assert_allclose(local[0],0,atol=1e-7);new,vv,mm=perturb_nominal(local,times,v0,p.config)
 np.savez_compressed(out/'controlled_nominal_probe.npz',raw_reference=raw,executed=executed,corrected_local=local,candidates=new,valid=vv);r=dict(label_access=False,initial_reference_displacement_m=float(np.linalg.norm(poses[0,:2])),old_valid=int(valid.sum()),controlled_valid=int(vv.sum()),controlled_manifest=mm,controller='pinned native 8s controller, preserves planned-reference initial origin for reference-speed derivation; actual ego initial state for execution');(out/'controlled_result.json').write_text(json.dumps(r,indent=2)+'\n');print('CONTROLLED_NOMINAL',r,flush=True)
if __name__=='__main__':main()
