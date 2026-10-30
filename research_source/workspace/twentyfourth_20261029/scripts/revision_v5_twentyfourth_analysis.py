"""R14 paired drive-cluster analysis; population identity comes from frozen manifest."""
from revision_v5_twentyfourth_common import *
STUDY=BASE
def cohort(scope):return BASE/scope,STUDY/'validation'/scope/'plan.json'
import numpy as np
from revision_v5_seventeenth_strict_aggregation import calculate_individual_mapping_scores,METRICS
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
import pandas as pd,argparse
EVENT={'NC':'no_at_fault_collisions','DAC':'drivable_area_compliance','DDC':'driving_direction_compliance','TLC':'traffic_light_compliance','TTC':'time_to_collision_within_bound','LK':'lane_keeping','HC':'history_comfort','EC':'two_frame_extended_comfort'}
def summary(tab,mapping):return calculate_individual_mapping_scores(tab[METRICS+['token','weight']],mapping)[0].to_dict()
def main():
    global BASE
    ap=argparse.ArgumentParser();ap.add_argument('scope');a=ap.parse_args();BASE=STUDY/'validation';out=BASE/a.scope;write(out/'plan.json',json.load(open(STUDY/'cohorts'/f'{a.scope}.json'))['plan']);res=json.load(open(out/'result.json'));assert res['complete'];plan=json.load(open(cohort(a.scope)[1]));groups=plan['groups'];rec={r['scene_id']:r for r in json.load(open(out/'choices.json'))};methods=list(res['methods']);tabs={m:pd.read_pickle(out/(m+'-scored.pkl')).set_index('token',drop=False).sort_index()for m in methods};mapping={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]for g in groups};gr=[];sc=[];events={};comparisons={}
    for m,t in tabs.items():
        t['drive']=[rec[s]['drive']for s in t.index];t['group']=[rec[s]['group']for s in t.index];t['candidate']=[rec[s]['choices'][m]for s in t.index];t['kind']=[rec[s]['origin']for s in t.index];t['method']=m
        for k in METRICS[:8]:t['physical_'+k]=[json.load(open(out/'scenes'/sid/f'candidate_{rec[sid]["choices"][m]}_physical.json'))['calls'][0]['raw'][k]for sid in t.index]
        keep=METRICS+['token','weight','drive','group','candidate','kind','method','pdm_score','endpoint_x','endpoint_y','executed_endpoint_x','executed_endpoint_y'];sc.append(t[keep+['physical_'+k for k in METRICS[:8]]])
        for g in groups:
            r=json.load(open(out/('group'+str(g['registered_group_index'])+'-'+m+'.json')))
            for stage,values in r.items():gr.append(dict(method=m,group=g['registered_group_index'],drive=g['drive'],stage=stage,**values))
        events[m]={domain:{e:int((t.loc[mask,col]<1).sum())for e,col in EVENT.items()}|dict(scenes=int(mask.sum()),hard_rule_union=int((t.loc[mask,['drivable_area_compliance','driving_direction_compliance','traffic_light_compliance']].min(axis=1)<1).sum()),P_below1=int((t.loc[mask,'ego_progress']<1).sum()))for domain,mask in [('all',np.ones(len(t),bool)),('original',t.kind=='original'),('synthetic',t.kind=='synthetic')]}
    groupdf=pd.DataFrame(gr);groupdf.to_csv(out/'per_mapping.csv',index=False);pd.concat(sc,ignore_index=True).to_csv(out/'per_scene.csv',index=False);groupdf.groupby(['method','drive','stage'])[METRICS].mean().reset_index().to_csv(out/'per_drive.csv',index=False)
    contrasts=[('PNC_H3P',r)for r in ['PNC_EC','PNC_NULL','IC0_H3P','Geometry_H3']]+[('PNC_NULL','PNC_EC')]
    for m,ref in contrasts:
        tm,tr=tabs[m],tabs[ref];assert list(tm.index)==list(tr.index);name=m+'_vs_'+ref;pair=tm[['token','drive','group','candidate','kind']].rename(columns={'candidate':'candidate_method'}).copy();pair['candidate_reference']=tr.candidate;pair['changed']=tm.candidate!=tr.candidate
        for k in METRICS+['weight']:pair[k+'_method']=tm[k];pair[k+'_reference']=tr[k];pair['delta_'+k]=tm[k]-tr[k]
        pair.to_csv(out/(name+'-paired_scenes.csv'),index=False);stats={}
        for stage in ['combined','stage_one','stage_two']:
            aa=groupdf[(groupdf.method==m)&(groupdf.stage==stage)].set_index('group');bb=groupdf[(groupdf.method==ref)&(groupdf.stage==stage)].set_index('group');delta=100*(aa.score-bb.score);d=aa.drive;dr=sorted(set(d));vals=np.array([delta[d==k].sum()for k in dr]);counts=np.array([(d==k).sum()for k in dr]);mean=vals/counts;rng=np.random.default_rng(102819);ix=rng.integers(len(dr),size=(10000,len(dr)));est=vals[ix].sum(1)/counts[ix].sum(1);driveboot=mean[ix].mean(1)
            stats[stage]=dict(delta_points=float(delta.mean()),CI95_mapping_equal_drive_cluster=np.quantile(est,[.025,.975]).tolist(),delta_drive_equal=float(mean.mean()),CI95_drive_equal=np.quantile(driveboot,[.025,.975]).tolist(),positive_drives=int((mean>1e-10).sum()),negative_drives=int((mean< -1e-10).sum()),zero_drives=int((abs(mean)<=1e-10).sum()),per_drive=dict(zip(dr,mean.tolist())),leave_one_drive_mapping_equal={k:float(delta[d!=k].mean())for k in dr},delta_P=float((aa.ego_progress-bb.ego_progress).mean()),largest_mapping_contribution_points=dict(group=int(delta.abs().idxmax()),contribution=float(delta.loc[delta.abs().idxmax()]/len(groups))))
        ev={};physical_events={}
        for e,col in EVENT.items():
            ae=tm[col]<1;be=tr[col]<1;ev[e]=dict(method_total=int(ae.sum()),reference_total=int(be.sum()),corrected=tm.index[be&~ae].tolist(),new=tm.index[ae&~be].tolist())
        for e,col in list(EVENT.items())[:-1]:
            ae=tm['physical_'+col]<1;be=tr['physical_'+col]<1;physical_events[e]=dict(method_total=int(ae.sum()),reference_total=int(be.sum()),corrected=tm.index[be&~ae].tolist(),new=tm.index[ae&~be].tolist())
        for prefix,target in [('',ev),('physical_',physical_events)]:
            rulecols=[prefix+k for k in ['drivable_area_compliance','driving_direction_compliance','traffic_light_compliance']]
            ae=tm[rulecols].min(axis=1)<1;be=tr[rulecols].min(axis=1)<1
            target['RULE_union']=dict(method_total=int(ae.sum()),reference_total=int(be.sum()),corrected=tm.index[be&~ae].tolist(),new=tm.index[ae&~be].tolist())
            for event,item in target.items():
                delta=np.array([int(t in item['new'])-int(t in item['corrected'])for t in tm.index])
                drives=sorted(set(tm.drive));sums=np.array([delta[tm.drive==d].sum()for d in drives]);counts=np.array([(tm.drive==d).sum()for d in drives]);rng=np.random.default_rng(102819);ix=rng.integers(len(drives),size=(10000,len(drives)))
                item['paired_rate_delta_percentage_points']=float(100*sums.sum()/counts.sum())
                item['CI95_scene_equal_drive_cluster_percentage_points']=np.quantile(100*sums[ix].sum(1)/counts[ix].sum(1),[.025,.975]).tolist()
        # Ordered exact arithmetic attribution, not causal attribution.
        # First replace all non-EC native terms and Q; then EC; finally weights.
        raw=pd.concat([pd.read_pickle(out/'scenes'/sid/f'candidate_{rec[sid]["choices"][m]}_filtered.pkl')for sid in tm.index],ignore_index=True).set_index('token').loc[tm.index];k=int(WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT);w=np.stack(raw.weighted_metrics_array);coef=raw.multiplicative_metrics_prod.to_numpy()*w[:,k]/w.sum(1)
        s1=tm.copy();s1['weight']=tr.weight;s1['score']=tm.score-coef*(tm.two_frame_extended_comfort-tr.two_frame_extended_comfort);s1['two_frame_extended_comfort']=tr.two_frame_extended_comfort
        s2=tm.copy();s2['weight']=tr.weight;v0=summary(tr,mapping)['score'];v1=summary(s1,mapping)['score'];v2=summary(s2,mapping)['score'];v3=summary(tm,mapping)['score'];decomp=dict(nonEC_terms_and_Q_under_reference_EC_weights=100*(v1-v0),EC_under_reference_endpoint_weights=100*(v2-v1),endpoint_weight_change=100*(v3-v2),total=100*(v3-v0),order_dependent=True,not_causal_effect_estimate=True);assert abs(decomp['total']-stats['combined']['delta_points'])<1e-8
        comparisons[name]=dict(statistics=stats,events=ev,physical_events=physical_events,choices_changed=int((tm.candidate!=tr.candidate).sum()),decomposition=decomp)
        if False: # original missing-mapping bounds remain historical, not rebranded confirmation
            delta=stats['combined']['delta_points'];dd=stats['combined']['delta_drive_equal']
            comparisons[name]['unmeasured_original_cohort_bounds']=dict(original17mapping_delta_points=[16/17*delta-100/17,16/17*delta+100/17],original10drive_delta_points=[.9*dd-10,.9*dd+10],meaning='algebraic worstcase usingmissingmapping0..100 eachmethod; notanestimate,CI,imputation orreason toreplaceeligible16mapping result')
    write(out/'analysis.json',dict(scope=a.scope,exposure=res['scope'],domain=DOMAIN,groups=len(groups),scenes=len(rec),drives=len({g['drive']for g in groups}),units='score/deltas x100;P0..1;events rawcounts component<1',events=events,comparisons=comparisons,bootstrap='10000 paired drivecluster seed102819; no independence assumption across synthetic branches',decomposition='ordered swaps nonEC/Q,EC,method endpointweights; stage products recomputed exactly; does not identify causal algorithm components'))
    print('ANALYSIS',{k:(v['statistics']['combined']['delta_points'],v['statistics']['combined']['CI95_mapping_equal_drive_cluster'],v['events']['NC'],v['events']['DAC'],v['statistics']['combined']['delta_P'])for k,v in comparisons.items()},flush=True)
if __name__=='__main__':main()
