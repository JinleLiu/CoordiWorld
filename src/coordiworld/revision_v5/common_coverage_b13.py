"""Separate physical planning-world control, never the adopted factual world.

Current road actors use absolute physical states. Non-road entities retain the
frozen factual model. No teacher/future labels at inference. Not independent
safety evidence: this simulator shares equations with the existing teacher.
"""
import numpy as np
import torch
from .compiled_solve_controller import predict_vector_controller_execution
from .current_rail_response import current_context
from .swept_template_features import extract_swept_rails
from .captured_factual_v2 import CapturedFactual,input_signature
from .absolute_planning_graph import CapturedAbsoluteSparse as CapturedSweptRail
from .conservative_rail_graph import build_graph
from .absolute_planning_graph import CapturedAbsoluteDense as DenseGraph


class CommonCoverageBaseWorld:
    def __init__(self,map_root,factual_model,pieces=4):
        self.map_root=map_root;self.factual=factual_model.eval().requires_grad_(False);self.graphs={};self.pieces=pieces;self.factual_graphs={}

    @torch.inference_mode()
    def predict(self, history, scene, tracks, factual_scene):
        """Inputs are current/history only; factual_scene has one reference action.

        response_available describes current rail coverage, not future validity
        or calibrated confidence. Unavailable response remains zero in the output.
        Invalid candidate outputs are returned with their explicit validity mask.
        """
        import navsim.traffic_agents_policies.navsim_IDM_traffic_agents as native
        from .idm_teacher_geometry import DOMAIN
        if getattr(native.build_ego_state_from_simulated_ego_states, '_coordiworld_teacher_domain', None) != DOMAIN:
            raise ValueError('Initialize pinned geometry before inference')
        cv = scene.candidate_valid[0].cpu().numpy()
        if len(cv) != 12 or not cv.any() or factual_scene.candidates.shape[1] != 1:
            raise ValueError('Expected twelve candidates and one factual reference')
        ref = int(np.flatnonzero(cv)[0])
        context = current_context(history, self.map_root)
        paths, info, tokens, initial = extract_swept_rails(context)
        execution, full, receipt = predict_vector_controller_execution(history, scene.candidates[0].cpu().numpy(), cv)
        origin = context.ego_state.rear_axle
        c, s = np.cos(origin.heading), np.sin(origin.heading)
        rotation = np.array([[c, s], [-s, c]])
        ego = np.zeros((12, 40, 4), np.float32)
        for j, values in full.items():
            v = values[:40]
            center = v[:, :2] + 1.461 * np.c_[np.cos(v[:, 2]), np.sin(v[:, 2])]
            ego[j, :, :2] = (center - np.array([origin.x, origin.y])) @ rotation.T
            ego[j, :, 2] = v[:, 2] - origin.heading
            ego[j, :, 3] = np.linalg.norm(v[:, 3:5], axis=-1)
        device = next(self.factual.parameters()).device
        lookup = {token: i for i, token in enumerate(tracks)}
        slots = torch.tensor([lookup.get(t, -1) for t in tokens], device=device, dtype=torch.long)
        good = slots >= 0
        x = factual_scene.to(device)
        factual_key=input_signature(x)
        if factual_key not in self.factual_graphs:self.factual_graphs[factual_key]=CapturedFactual(self.factual,x)
        reference=self.factual_graphs[factual_key](x)
        capacity = max(8, ((len(slots)+7)//8)*8)
        neighbors,nvalid,graph_stats=build_graph(paths,info,initial,capacity)
        sparse=graph_stats['neighbors']<capacity
        key = (capacity,graph_stats['neighbors'] if sparse else capacity, ref,sparse)
        if key not in self.graphs:
            self.graphs[key] = CapturedSweptRail(capacity,graph_stats['neighbors'], ref,pieces=self.pieces) if sparse else DenseGraph(capacity,ref,pieces=self.pieces)
        arguments=[torch.from_numpy(v).to(device) for v in [paths,info,ego,initial]]
        if sparse:arguments.extend([torch.from_numpy(neighbors).to(device),torch.from_numpy(nvalid).to(device)])
        d = self.graphs[key](*arguments)
        graph_stats['execution']='packed' if sparse else 'dense_no_pair_work_reduction'
        world = reference[None].expand(12,-1,-1,-1).clone()
        world[:, :, slots[good]] = d[:, :, good]
        world[..., 2] = torch.atan2(world[..., 2].sin(), world[..., 2].cos())
        delta = world-world[ref:ref+1]
        delta[...,2] = torch.atan2(delta[...,2].sin(),delta[...,2].cos())
        available = torch.zeros(scene.agents.shape[2], dtype=torch.bool, device=device)
        available[slots[good]] = True
        return {'execution': execution, 'execution_full': full, 'full_rail_states': d, 'full_rail_tokens': tokens, 'states': world, 'delta': delta, 'reference_states': world[ref], 'unchanged_factual_reference': reference,
                'reference': ref, 'candidate_valid': cv,
                'response_available': available, 'input_receipt': receipt,
                'rail_actor_count': len(tokens),'interaction_graph':graph_stats, 'qualified': False,
                'full_EPDMS': None, 'future_validity_calibrated': False,'response_is_learned':False,'factual_execution_graph':True,'static_map_index':True,'prediction_cache':False,'velocity_semantics':'instantaneous legacy-IDM paired response','heading_semantics':'wrapped paired body heading','kinematic_consistency_independently_proven':False,'absolute_teacher_domain_planning_diagnostic':True,'preserves_factual_reference_in_this_arm':False,'changes_default_factual_world':False}


def common_reference_world(history, scene, tracks, prediction, execution):
    """Expose the SAME current entity set to geometric and learned readouts.

    No observed current actor is discarded merely for being farther than50m.
    Keeping all currently supplied annotations is conservative with respect to
    the4s candidate envelope. The reachability flag below is descriptive only:
    a2m/s^2 acceleration envelope is not a guaranteed exclusion certificate.
    This does not invent actors outside the observation source's own coverage.
    Physical rail membership and interaction/order are unchanged from B11.
    New unsupported actors receive explicit CV, not an implicit safe label.
    """
    from types import SimpleNamespace
    from coordiworld.data.openscene import frame_agents, ego_pose
    current=frame_agents(history[-1],ego_pose(history[-1]))
    ref=prediction['reference']; old=prediction['states'][ref]
    n=old.shape[1]; rail_by={t:i for i,t in enumerate(prediction['full_rail_tokens'])}
    extra=sorted((set(current)|set(rail_by))-set(tracks))
    worlds=[old]; validity=[scene.agent_valid[0,-1].to(old.device)]
    ids=list(tracks)+['__invalid_padding_'+str(i) for i in range(len(tracks),n)]
    sources=['physical' if t in rail_by else 'frozen_factual' for t in tracks]+['invalid']*(n-len(tracks))
    states=[]; receipts=[]; time=old.new_tensor(np.arange(1,9)*.5)
    cv=scene.candidate_valid[0].cpu().numpy(); candidate_xy=np.asarray(execution)[cv,:,:2]
    for token in extra:
        if token in rail_by:
            w=prediction['full_rail_states'][ref,:,rail_by[token]].clone()
            w[:,2]=torch.atan2(w[:,2].sin(),w[:,2].cos()); source='physical'
        else:
            state=old.new_tensor(current[token][0]);w=state[None].expand(8,-1).clone()
            w[:,:2]+=time[:,None]*state[None,3:5];source='current_CV_unsupported'
        worlds.append(w[:,None]);validity.append(torch.ones(1,dtype=torch.bool,device=old.device));ids.append(token);sources.append(source)
        raw=current.get(token);state=None if raw is None else raw[0]
        distance=None if state is None else float(np.linalg.norm(candidate_xy-state[:2],axis=-1).min())
        bound=None if state is None else float(4*np.linalg.norm(state[3:5])+16+.5*np.linalg.norm(state[5:7])+3)
        receipts.append(dict(token=token,source=source,current_state=None if state is None else state.tolist(),minimum_distance_to_candidate_samples_m=distance,descriptive4s_reach_bound_m=bound,inside_descriptive_envelope=distance is None or distance<=bound))
    world=torch.cat(worlds,1);valid=torch.cat(validity)
    proxy=SimpleNamespace(ego=scene.ego.to(old.device),ego_rear_axle_to_center=scene.ego_rear_axle_to_center.to(old.device),map_valid=scene.map_valid.to(old.device),agent_valid=valid[None,None])
    assert torch.equal(world[:,:n],old)
    return proxy,world,dict(reference=ref,original_slots=n,original_valid=int(validity[0].sum()),current_observed=len(current),added=len(extra),common_valid=int(valid.sum()),common_tokens=ids,sources=sources,added_entities=receipts,retained_states_exact=True,scope='All current supplied finite entities; no outcome inputs; new nonrail fallback CV explicitly unsupported; no beyond-source coverage claim')
