"""Resolve the real B36 original roster, without inference or outcome access.

This is NOT a fabricated official two-stage cohort. Its input rows can be fed
to Pipeline.inputs / original-scene native scoring; synthetic mappings remain
a separate missing dependency.
"""
from revision_v5_twentyfourth_common import *
import pickle,numpy as np

def main():
    meta=json.load(open(BASE/'cohorts/B36_authoritative.json'))
    groups={}
    for r in meta['rows']:groups.setdefault(r['path'],[]).append(r)
    rows=[];failures=[];assets={}
    for relative,rr in sorted(groups.items()):
        path=ROOT/relative
        if not path.exists():
            failures.append(dict(path=str(path),reason='missing original log'));continue
        digest=sha(path);assets[str(path)]=digest
        assert all(r['sha256']==digest for r in rr),(path,'roster raw byte mismatch')
        with path.open('rb')as f:frames=pickle.load(f)
        for r in rr:
            j=r['index'];current=frames[j]
            assert str(current['token'])==r['token'] and int(current['timestamp'])==r['timestamp']
            h=frames[j-3:j+1]
            good=j>=3 and j+8<len(frames) and len(h)==4 and np.allclose(np.diff([f['timestamp']for f in h])/1e6,.5,atol=.08)
            row=dict(scene_id=r['token'],drive=r['drive'],origin='original',source=dict(server_path=str(path),sha256=digest,current_index=j,log_name=r['log'],timestamp_us=r['timestamp']),
                current_history_indices=list(range(j-3,j+1)),native_future_end_index=j+8,input_and_native_asset_ready=bool(good),
                legal_previous_history_ready=bool(j>=4 and round((current['timestamp']-frames[j-1]['timestamp'])/1e5)in range(1,6)),
                route='original-scene interface only; no synthetic mapping is inferred from adjacency')
            if not good:failures.append(dict(scene_id=r['token'],reason='history/native asset contract'))
            rows.append(row)
    exposure=json.load(open(ROOT/'nineteenth_20261028/EXPOSURE_SETS.json'))
    world=json.load(open(ROOT/'nineteenth_20261028/WORLD_EXPOSURE.json'))
    drives=set(meta['drives']);world_drives=set(world['assigned_fit_drives'])|set(world['assigned_probe_drives'])
    dev={r['drive']for scope in ['full140','exposed408']for r in json.load(open(BASE/'cohorts'/f'{scope}.json'))['rows']}
    result=dict(created_utc=now(),unit='36 drives / 119 logs / 1280 original scene anchors',authoritative_roster_sha=meta['source_sha'],
        requested_scenes=1280,ready_scenes=sum(r['input_and_native_asset_ready']for r in rows),rows=rows,asset_sha256=assets,failures=failures,
        known_exposure_audit=dict(B_head_overlap=sorted(drives&set(exposure['B_head_train_drives'])),B_world_fit_or_probe_overlap=sorted(drives&world_drives),B_full_development_overlap=sorted(drives&dev),
            qualification='known recorded B lineage only; this metadata audit does not assert geographic independence or invent absent A training provenance'),
        original_input_interface='revision_v5_twentyfirst_full.history(row, logs) -> four current/history frames; corrected Pipeline.inputs(history, source.sha256)',
        official_full_protocol_ready=False,missing=meta['missing'],inference_executed=False,outcome_labels_read=False,
        reason_not_run='R14 prioritizes the verified complete protected20 cohort. Original-only B36 scores cannot replace its missing official two-stage protocol.')
    write(BASE/'cohorts/B36_original_inputs_ready.json',result)
    print('B36_INPUTS_READY',len(rows),result['ready_scenes'],len(assets),'missing',len(failures),result['known_exposure_audit'],flush=True)
if __name__=='__main__':main()
