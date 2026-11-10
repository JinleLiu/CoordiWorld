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


class SweptOccupancyHybridWorld:
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
        _, full, receipt = predict_vector_controller_execution(history, scene.candidates[0].cpu().numpy(), cv)
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
        return {'states': world, 'delta': delta, 'reference_states': world[ref], 'unchanged_factual_reference': reference,
                'reference': ref, 'candidate_valid': cv,
                'response_available': available, 'input_receipt': receipt,
                'rail_actor_count': len(tokens),'interaction_graph':graph_stats, 'qualified': False,
                'full_EPDMS': None, 'future_validity_calibrated': False,'response_is_learned':False,'factual_execution_graph':True,'static_map_index':True,'prediction_cache':False,'velocity_semantics':'instantaneous legacy-IDM paired response','heading_semantics':'wrapped paired body heading','kinematic_consistency_independently_proven':False,'absolute_teacher_domain_planning_diagnostic':True,'preserves_factual_reference_in_this_arm':False,'changes_default_factual_world':False}
