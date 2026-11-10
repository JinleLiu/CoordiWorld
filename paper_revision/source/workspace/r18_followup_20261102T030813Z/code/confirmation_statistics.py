"""Preregistered paired route-cluster estimates; no model or simulator dependencies."""
import random,math
from collections import defaultdict
def quantile(values,p):
    if not values:return None
    x=sorted(values);i=(len(x)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return x[lo]+(x[hi]-x[lo])*(i-lo)
def contrast(rows,routes,seeds,arm_a,arm_b,common_records=False,aligned_pairs=None):
    index={(r['route'],r['seed'],r['method']):r for r in rows};assert len(index)==len(rows)
    strata={r['stratum']:[]for r in routes};effects=[];excluded=[]
    for r in routes:
        values=[];eligible=True
        for seed in seeds:
            a=index[(r['id'],seed,arm_a)];b=index[(r['id'],seed,arm_b)]
            assert a['DS']is not None and b['DS']is not None and math.isfinite(a['DS'])and math.isfinite(b['DS'])
            if common_records and not(a['official_record_present']and b['official_record_present']):eligible=False
            if aligned_pairs is not None and(r['id'],seed)not in aligned_pairs:eligible=False
            values.append(a['DS']-b['DS'])
        if not eligible:excluded.append(r['id']);continue
        effect=sum(values)/len(seeds);strata[r['stratum']].append(effect);effects.append(dict(route=r['id'],stratum=r['stratum'],seed_differences=values,mean=effect))
    counts={k:len(v)for k,v in strata.items()};result=dict(direction=arm_a+' minus '+arm_b,route_count=len(effects),paired_seed_count=len(effects)*len(seeds),stratum_counts=counts,excluded_routes=excluded,route_effects=effects,mean=None,CI95=None,CI97_5=None,bootstrap_samples=10000,bootstrap_seed=20261102,interpretation='sensitivity changes population; never replaces all-scheduled primary'if common_records or aligned_pairs is not None else'all scheduled cells, missing official score0')
    if not all(strata.values()):result['unavailable_reason']='at least one prespecified stratum has no complete route';return result
    assert len(strata)==3
    result['mean']=sum(sum(v)/len(v)for v in strata.values())/3
    rng=random.Random(20261102);boot=[]
    for _ in range(10000):boot.append(sum(sum(v[rng.randrange(len(v))]for _ in v)/len(v)for v in strata.values())/3)
    result['CI95']=[quantile(boot,.025),quantile(boot,.975)];result['CI97_5']=[quantile(boot,.0125),quantile(boot,.9875)]
    result['limited_route_support']=any(n<2 for n in counts.values());return result
def self_test():
    routes=[dict(id=str(i),stratum=str(i%3))for i in range(12)];seeds=[17,23,42];rows=[dict(route=r['id'],seed=s,method=m,DS=float(int(r['id'])+s+(2 if m=='A'else 0)),official_record_present=True)for r in routes for s in seeds for m in ['A','B']]
    x=contrast(rows,routes,seeds,'A','B');y=contrast(rows,routes,seeds,'B','A');assert x['mean']==2 and x['CI97_5']==[2,2]and y['mean']==-2 and x['route_count']==12 and x['paired_seed_count']==36
    rows[0]['official_record_present']=False;rows[0]['DS']=0
    assert contrast(rows,routes,seeds,'A','B')['route_count']==12
    assert contrast(rows,routes,seeds,'A','B',common_records=True)['route_count']==11
    aligned={(r['id'],s)for r in routes if r['stratum']!='0'for s in seeds};assert contrast(rows,routes,seeds,'A','B',aligned_pairs=aligned)['mean']is None
    return dict(passed=True,checks=['paired direction','three seeds remain one route cluster','constant shift CI','scheduled missing-record denominator retained','complete-route sensitivity','missing stratum not renormalized'])
if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
