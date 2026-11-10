"""CUDA graph of full-state physics with conservatively packed interactions."""
import torch
from .sparse_swept_rollout import rollout


class CapturedSparseSweptFullState:
    def __init__(self, capacity, neighbors, reference=0, iterations=2,pieces=4):
        self.capacity=capacity;self.reference=reference;self.iterations=iterations;self.pieces=pieces
        self.paths=torch.full((capacity,256,3),1e6,device='cuda');self.info=torch.zeros(capacity,8,device='cuda');self.info[:,1]=1;self.info[:,5]=10
        self.ego=torch.zeros(12,40,4,device='cuda');self.initial=torch.full((capacity,3),1e6,device='cuda')
        self.neighbors=torch.zeros(capacity,neighbors,device='cuda',dtype=torch.long)
        self.neighbor_valid=torch.zeros(capacity,neighbors,device='cuda',dtype=torch.bool)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(2):self._forward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph),torch.inference_mode():self.output=self._forward()
        torch.cuda.synchronize()

    def _forward(self):
        states,_=rollout(self.paths,self.info,self.ego,self.initial,self.neighbors,self.neighbor_valid,iterations=self.iterations,pieces=self.pieces)
        delta=states-states[self.reference:self.reference+1]
        heading=torch.atan2(delta[...,2].sin(),delta[...,2].cos())
        return torch.cat((delta[...,:2],heading[...,None],delta[...,3:5],torch.zeros_like(delta[...,5:7])),-1)

    @torch.inference_mode()
    def __call__(self,paths,info,ego,initial,neighbors,neighbor_valid):
        n=len(info)
        if n>self.capacity or ego.shape!=(12,40,4) or neighbors.shape!=self.neighbors.shape:raise ValueError('Graph shape exceeded')
        self.paths.fill_(1e6);self.info.zero_();self.info[:,1]=1;self.info[:,5]=10;self.initial.fill_(1e6)
        self.paths[:n].copy_(paths);self.info[:n].copy_(info);self.ego.copy_(ego);self.initial[:n].copy_(initial)
        self.neighbors.copy_(neighbors);self.neighbor_valid.copy_(neighbor_valid);self.graph.replay()
        return self.output[:,:,:n]
