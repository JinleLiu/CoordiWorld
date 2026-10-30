"""B17 live current-only B features for frozen full-mapping interface checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2');os.environ.setdefault('NUPLAN_MAPS_ROOT',__import__('coordiworld_paths').workspace_path('data/maps'))
import json,copy
from pathlib import Path
import numpy as np,torch
import revision_v5_eleventh_new_scenes_progress as b
from coordiworld.data.openscene import make_inputs,frame_agents,ego_pose
from coordiworld.data.pdm_proposer import PDMNominalProposer
from coordiworld.revision_v5.common_coverage_b13 import CommonCoverageBaseWorld,common_reference_world
from coordiworld.revision_v5.current_history_comfort import current_history_comfort
from coordiworld.revision_v5.component_known_b13 import from_legacy,partial_current_DAC,geometry_known_score
from revision_v5_sixteenth_semantics import semantic_features,current_areas
from revision_v5_seventeenth_progress import Extractor,p_context
from revision_v5_seventeenth_train import predictions
from revision_v5_sixteenth_common import pair_features,choose,sha
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));BASE=ROOT/'seventeenth_20261027';OLD=ROOT/'sixteenth_20261027'
class Pipeline:
    def __init__(self,config):
        # Match the frozen B16 x86 execution contract even when NumPy was
        # imported before environment variables. This does not change S64.
        from threadpoolctl import threadpool_limits
        self.thread_limit=threadpool_limits(limits=1)
        from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
        install_corrected_builder();self.config=config;self.provider=b.CurrentMapProvider(ROOT/'data/maps');self.proposer=PDMNominalProposer(ROOT/'data/maps',config['candidates']);self.progress=Extractor()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.recompile_limit=128
        cp=ROOT/'runs/dynamics_residual_batch_v1/kinematic_residual_s17/unqualified_model.pt';ck=torch.load(cp,map_location='cpu',weights_only=True);model=b.BodyHeadingKinematicResidual(b.ModelConfig(**ck['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state_dict']);self.world=CommonCoverageBaseWorld(ROOT/'data/maps',model);self.checkpoint_sha256=sha(cp)
    @torch.inference_mode()
    def inputs(self,history,source_sha,scene=None,meta=None):
        now=history[-1];origin=np.asarray(ego_pose(now));cur=frame_agents(now,origin)
        if scene is None:
            config=copy.deepcopy(self.config);config['data']['max_agents']=max(1,sum(np.linalg.norm(s[:2])<=config['data']['roi_m']for s,_ in cur.values()));scene,meta=make_inputs(history,config,self.provider(history,config['data']['max_map_tokens']),self.proposer(history));assert meta['truncated_agents']==0
        pred=self.world.predict(history,scene,meta['tracks'],b.factual_inputs(scene));exe=pred['execution'];full=pred['execution_full'];valid=scene.candidate_valid[0].numpy();ids=np.flatnonzero(valid);ref=int(valid.argmax());proxy,world,cov=common_reference_world(history,scene,meta['tracks'],pred,exe)
        mm,rv=b.execution_rules(scene,meta,history,exe,self.provider);cf=b.candidate_features(proxy,torch.from_numpy(exe).cuda(),torch.from_numpy(mm).cuda(),torch.from_numpy(rv).cuda(),world[None].expand(12,-1,-1,-1)).cpu().numpy()
        c,s=np.cos(origin[2]),np.sin(origin[2]);rotation=np.array([[c,-s],[s,c]]);local=np.zeros((12,41,11))
        for j,values in full.items():
            local[j]=values;local[j,:,:2]=(values[:,:2]-origin[:2])@rotation;local[j,:,2]=np.arctan2(np.sin(values[:,2]-origin[2]),np.cos(values[:,2]-origin[2]));local[j,:,3:5]=values[:,3:5]@rotation;local[j,:,5:7]=values[:,5:7]@rotation
        dyn=np.asarray(now['ego_dynamic_state']);current=dict(scene_id=str(now['token']),timestamp_us=int(now['timestamp']),latest_source_timestamp_us=int(now['timestamp']),map_name=now['map_location'],rear_axle_pose=origin.tolist(),velocity=dyn[:2].tolist(),acceleration=dyn[2:].tolist(),route_roadblock_ids=list(now['roadblock_ids']),traffic_lights=[list(v)for v in now['traffic_lights']],source_container_sha256=source_sha)
        hc,hcr=current_history_comfort(history,full,ROOT/'data/maps');geo=np.zeros((12,18),np.float32);map_error=None;bundle=None
        try:
            bundle,_,_=b.export_current_map(current,ROOT/'data/maps',local[ids,:,:3]);e,t=b.build_trace(bundle,local[ids,:,:3]);geo[ids]=b.static_features(e,t)
        except Exception as er:map_error=repr(er)
        known=from_legacy(geo,hc)
        if not known[:,3].any()and bundle is not None:
            dac,flags=partial_current_DAC(bundle,local[ids,:,:3]);known[ids,0]=flags['map_geometry_available'];known[ids,1]=flags['execution_envelope_available'];known[ids,2]=flags['route_resolved'];kn=flags['map_geometry_available']and flags['execution_envelope_available'];known[ids,3]=kn;known[ids,4]=dac
            if flags['route_resolved']and kn:known[ids,5]=geo[ids,0];known[ids,6]=geo[ids,2];known[ids,7]=geo[ids,0];known[ids,8]=geo[ids,4]
        f198=np.concatenate([cf,known],-1);gs=geometry_known_score(f198[None],known[None],valid[None]);g=int(choose(gs,valid[None])[0])
        tokens=cov['common_tokens'];initial=np.zeros((len(tokens),7),np.float32);ck=np.zeros(len(tokens),bool);types=np.full(len(tokens),-1,np.int64)
        for j,token in enumerate(tokens):
            if token in cur:initial[j]=cur[token][0];types[j]=cur[token][1];ck[j]=True
            elif j<len(meta['tracks']):initial[j]=scene.agents[0,-1,j].numpy();ck[j]=bool(scene.agent_valid[0,-1,j])
        assets=dict(world=world.cpu().numpy(),actor_valid=proxy.agent_valid[0,-1].cpu().numpy(),current=initial,current_known=ck,types=types,tokens=np.array(tokens),sources=np.array(cov['sources']),execution=exe,execution_full_local=local,candidate_valid=valid,ego_box=scene.ego[0,-1,5:7].numpy(),rear_axle_offset=scene.ego_rear_axle_to_center.numpy());area,ainfo=current_areas(current,local,ROOT/'data/maps');s64,_,trace=semantic_features(assets,area);pknown=True;p_error=None;pinfo={}
        try:pa,pinfo=self.progress.extract(history,assets);dist,pd,remaining=pa['distance'],pa['pdm_distance'],pa['remaining']
        except Exception as er:pknown=False;p_error=repr(er);dist=np.zeros(12);pd=0;remaining=np.zeros(12);pa={}
        p16,p0=p_context(dist,pd,remaining,pknown,np.where(known[:,7],known[:,8],0),known[:,7].astype(bool)&valid,valid,ref,g);xo=np.concatenate([pair_features(f198[None],valid[None]),s64[None]],-1).astype(np.float32);xx=np.concatenate([xo,p16[None]],-1).astype(np.float32)
        return dict(scene_id=str(now['token']),reference=ref,Geometry_HC=g,valid=valid,candidates=scene.candidates[0].numpy(),features198=f198,S64=s64,P16=p16,p0=p0,xold=xo,x=xx,assets=assets,areas=area,pdm_assets=pa,metadata=meta,current=current,coverage=cov,HC_receipt=hcr,map_error=map_error,PDM_error=p_error,PDM_identity=pinfo,S64_trace=trace)
    def select(self,a):
        freeze=json.load(open(BASE/'PRIMARY_FREEZE.json'));recipe=freeze['recipe'];root=BASE/'runs'/recipe['id']/'final/fit';binding=json.load(open(root/'bindings.json'));models={h:(Path(v['path']),v['old_input'])for h,v in binding.items()};v=a['valid'][None];geo=np.array([a['Geometry_HC']]);pred,ms=predictions(models,a['x'],a['xold'],recipe['qp_mode'],a['p0'][None],a['features198'][None,:,197],v,geo)
        old={h:(OLD/'runs/J4SR/final/fit'/h/'model.ubj',True)for h in ['q','qp','qttc','qlk']};op,oms=predictions(old,a['x'],a['xold'],'PD',a['p0'][None],a['features198'][None,:,197],v,geo)
        return dict(Geometry_HC=a['Geometry_HC'],B16_J4SR=int(op['choice'][0]),B17_primary=int(pred['choice'][0])),dict(B17_primary=pred,B16_J4SR=op),dict(B17=ms,B16=oms)
