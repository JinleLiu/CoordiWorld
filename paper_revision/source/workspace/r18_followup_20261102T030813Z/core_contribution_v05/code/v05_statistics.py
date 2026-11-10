"""Frozen v05 route-equal, stratified paired-cluster analysis; no seed pseudoreplication."""
import random,math

def quantile(values,p):
 if not values:return None
 a=sorted(values);q=(len(a)-1)*p;i=int(q);return a[i]*(1-(q-i))+a[min(i+1,len(a)-1)]*(q-i)

def contrast(rows,routes,seeds,a,b,repetitions=50000,seed=20261105):
 idx={(r['route'],r['seed'],r['method']):r for r in rows};effects=[]
 for route in routes:
  rr=route['id'];aa=[idx[rr,s,a]['DS']for s in seeds];bb=[idx[rr,s,b]['DS']for s in seeds]
  effects.append(dict(route=rr,stratum=route['stratum'],seeds=seeds,arm_a_mean=sum(aa)/len(seeds),arm_b_mean=sum(bb)/len(seeds),DS_difference=sum(x-y for x,y in zip(aa,bb))/len(seeds),seed_differences=[x-y for x,y in zip(aa,bb)]))
 strata={s:[r['DS_difference']for r in effects if r['stratum']==s]for s in sorted({r['stratum']for r in effects})}
 rng=random.Random(seed);boot=[]
 for _ in range(repetitions):boot.append(sum(rng.choice(v)for v in strata.values()for _ in v)/len(effects))
 delta=sum(r['DS_difference']for r in effects)/len(effects)
 return dict(direction=a+' minus '+b,DS_difference=delta,CI95=[quantile(boot,.025),quantile(boot,.975)],route_count=len(effects),seed_count=len(seeds),route_effects=effects,stratum_effects={s:sum(v)/len(v)for s,v in strata.items()},bootstrap_repetitions=repetitions,bootstrap_seed=seed,bootstrap_unit='paired route cluster resampled within fixed strata; all seeds/arms retained together',positive_routes=sum(r['DS_difference']>0 for r in effects),negative_routes=sum(r['DS_difference']<0 for r in effects),zero_routes=sum(r['DS_difference']==0 for r in effects),working_increment_reference=2.,working_increment_met=delta>=2.,not_equivalence_test=True)

def self_test():
 routes=[dict(id=str(i),stratum='a'if i<4 else'b'if i<6 else'c')for i in range(8)];rows=[]
 for r in routes:
  for s in [17,23,42]:
   for m,v in [('H-Map',6),('H-Generic',3)]:rows.append(dict(route=r['id'],seed=s,method=m,DS=v))
 x=contrast(rows,routes,[17,23,42],'H-Map','H-Generic',1000);assert x['DS_difference']==3 and x['CI95']==[3,3]
 for r in rows:
  if r['method']=='H-Map':r['DS']=3+(8 if r['route']=='0'else 0)
 x=contrast(rows,routes,[17,23,42],'H-Map','H-Generic',1000);assert x['DS_difference']==1
 y=contrast(rows,routes,[17,23,42],'H-Generic','H-Map',1000);assert y['DS_difference']==-1 and abs(y['CI95'][0]+x['CI95'][1])<1e-12
 assert len(x['route_effects'])==8 and x['route_effects'][0]['seed_differences']==[8]*3
 return dict(passed=True,checks=['constant effect CI','eight equal route weights','seed triplets remain one cluster','paired direction reversal','stratum sizes retained'])
if __name__=='__main__':
 import json;print(json.dumps(self_test()))
