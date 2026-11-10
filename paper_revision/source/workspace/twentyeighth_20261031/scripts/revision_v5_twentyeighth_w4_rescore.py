"""Official 7814310e scoring equations, reused on the entire augmented pool.

Borrowed: navsim/agents/drivoR/drivor_model.py forward scoring block.
Cross-candidate attention is recomputed; image/ego context is frozen per call.
"""
import torch
class FrozenPoolScorer:
 def __init__(self,model):
  self.model=model;self.context=None;self.ego=None
  def ctx(mod,args):self.context=args[1]
  def ego(mod,args,out):self.ego=out[:,None]
  self.hooks=[model.future_predictor.register_forward_pre_hook(ctx),model.hist_encoding.register_forward_hook(ego)]
 @torch.no_grad()
 def score(self,proposals):
  assert self.context is not None and self.ego is not None
  m=self.model;p=torch.as_tensor(proposals,device=self.context.device,dtype=self.context.dtype)[None];b,n,_,_=p.shape;e=m.pos_embed(p.reshape(b,n,-1).detach());tr=m.future_predictor(e,self.context)+self.ego;logits=m.scorer(p,tr)[0];c=m._config
  score=(c.noc*logits['no_at_fault_collisions'].sigmoid().log()+c.dac*logits['drivable_area_compliance'].sigmoid().log()+c.ddc*logits['driving_direction_compliance'].sigmoid().log()+(c.ttc*logits['time_to_collision_within_bound'].sigmoid()+c.ep*logits['ego_progress'].sigmoid()+c.comfort*logits['comfort'].sigmoid()).log())
  return score[0].float().cpu().numpy(),{k:v[0].float().cpu().numpy()for k,v in logits.items()}
 def close(self):
  for h in self.hooks:h.remove()
