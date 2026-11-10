"""Metadata-only causal input coverage, including the inherited R18 oldest-slot adapter."""
import json
import pickle
from pathlib import Path
import numpy as np
from a0_train_source_audit import RUN, ROOT, write, now


def history_kind(raw, i):
    if i < 3:
        return 'missing_four_observed_frames'
    times=np.array([int(x['timestamp'])for x in raw[i-3:i+1]],dtype=np.int64)
    if np.allclose(np.diff(times)/1e6,.5,atol=.08):
        return 'original_regular_four'
    if not np.all(np.diff(times)>0) or not np.allclose(np.diff(times[-3:])/1e6,.5,atol=.08):
        return 'unsupported_current_or_previous_gap'
    targets=times[-1]-np.arange(3,-1,-1,dtype=np.int64)*500000
    if any(int(np.argmin(abs(times-targets[k])))!=k or abs(int(times[k]-targets[k]))>80000 for k in range(1,4)):
        return 'unsupported_retained_slot_timing'
    if abs(times[np.argmin(abs(times-targets[0]))]-targets[0])<=80000:
        return 'unsupported_nonmissing_oldest_slot'
    lo=np.flatnonzero(times<targets[0]);hi=np.flatnonzero(times>targets[0])
    if not len(lo)or not len(hi)or times[int(hi[0])]-times[int(lo[-1])]>1100000:
        return 'unsupported_ego_extrapolation_or_long_gap'
    return 'inherited_R18_causal_oldest_slot_adapter'


def main():
    rows=json.loads((RUN/'evidence/TRAIN_SOURCE_INDEX.json').read_text())['rows']
    cfg=json.loads((RUN/'configs/DATA_ALIGNMENT_FREEZE_v01.json').read_text())
    w4set=set(sum(cfg['w4_pilot_batches'],[]));bylog={}
    for row in rows:bylog.setdefault(row['raw_log'],[]).append(row)
    logs=[];coverage=[];excluded=[];camera_files={};camera_scenes=[]
    allowed={'original_regular_four','inherited_R18_causal_oldest_slot_adapter'}
    for path,targets in bylog.items():
        raw=pickle.loads(Path(path).read_bytes());target_by_index={x['current_index']:x for x in targets}
        indices=sorted({j for i in target_by_index for j in range(max(3,i-3),i+1)})
        kinds={i:history_kind(raw,i)for i in indices};segments=[]
        for i in indices:
            if kinds[i]not in allowed:
                excluded.append(dict(token=str(raw[i]['token']),log=path,index=i,reason=kinds[i],natural_target=i in target_by_index));continue
            if not segments or i!=segments[-1][-1]['index']+1 or not 0<int(raw[i]['timestamp'])-segments[-1][-1]['timestamp_us']<550000:
                segments.append([])
            segments[-1].append(dict(index=i,token=str(raw[i]['token']),timestamp_us=int(raw[i]['timestamp']),
                                    input_kind=kinds[i],natural_target=i in target_by_index))
        for segment in segments:
            for j,node in enumerate(segment):
                if node['natural_target']:
                    coverage.append(dict(token=node['token'],drive=targets[0]['drive'],fold=targets[0]['fold'],
                                         input_kind=node['input_kind'],has_legal_previous_plan=j>0,
                                         previous_input_kind=segment[j-1]['input_kind']if j else None,
                                         segment_start_token=segment[0]['token']))
        logs.append(dict(raw_log=path,raw_log_sha256=targets[0]['raw_log_sha256'],drive=targets[0]['drive'],fold=targets[0]['fold'],segments=segments))
        chosen=[x['current_index']for x in targets if x['token']in w4set]
        native_indices=sorted({j for i in chosen for j in range(max(3,i-3),i+1)})
        for i in native_indices:
            for cam in ['CAM_F0','CAM_L0','CAM_R0','CAM_B0']:
                rel=str(raw[i]['cams'][cam]['data_path'])
                options=[ROOT/'A_R18-research-20261031T1047/sensors'/rel,ROOT/'data/recovery/raw/sensors'/rel,ROOT/'data/recovery/raw/sensor_blobs/trainval'/rel]
                camera_files[rel]=dict(path=rel,found_at=next((str(p)for p in options if p.is_file()),None))
        camera_scenes.extend(x['token']for x in targets if x['token']in w4set)
    write('configs/CAUSAL_INFERENCE_MANIFEST_v01.json',dict(created_at=now(),natural_roster_unchanged=True,
          inherited_adapter=str(ROOT/'A_R18-research-20261031T1047/code/pnc_missing_history_slots_v02.py'),
          adapter_semantics='Only oldest actor slot may be UNKNOWN; ego interpolation is bracketed by already observed past states; no observed timestamp relabeled, no future/current extrapolation. Original regular entry unchanged.',
          inactive_prefix_rule='Unsupported prefix nodes cannot generate plans; reset at the next supported node. Retain every requested target in coverage/missing accounting.',logs=logs))
    write('evidence/CAUSAL_CONTEXT_COVERAGE.json',dict(at=now(),requested_targets=1755,targets_with_legal_current=len(coverage),
          targets_with_legal_previous=sum(x['has_legal_previous_plan']for x in coverage),
          previous_uses_inherited_oldest_slot_adapter=sum(x['previous_input_kind']=='inherited_R18_causal_oldest_slot_adapter'for x in coverage),
          inference_records=sum(len(s)for log in logs for s in log['segments']),
          original_regular_records=sum(n['input_kind']=='original_regular_four'for l in logs for s in l['segments']for n in s),
          inherited_adapter_records=sum(n['input_kind']=='inherited_R18_causal_oldest_slot_adapter'for l in logs for s in l['segments']for n in s),
          excluded_records=excluded,coverage=coverage,new_outcomes_read=False))
    write('evidence/W4_EXACT_SENSOR_REQUIREMENTS.json',dict(at=now(),source='Original fixed drivoR.yaml and get_sensor_config',
          cameras=['CAM_F0','CAM_L0','CAM_R0','CAM_B0'],consumed_history_index=[3],lidar=False,
          frozen_targets=len(camera_scenes),unique_required_images=len(camera_files),
          found_images=sum(x['found_at']is not None for x in camera_files.values()),files=list(camera_files.values()),
          correction='The earlier4480 figure was a conservative all-camera/all-raw-history envelope, not the actual native model input count.',new_inference=False))
    print(json.dumps(dict(targets_with_legal_current=len(coverage),targets_with_legal_previous=sum(x['has_legal_previous_plan']for x in coverage),
          inherited_adapter_records=sum(n['input_kind']=='inherited_R18_causal_oldest_slot_adapter'for l in logs for s in l['segments']for n in s),
          W4_exact_required_images=len(camera_files),W4_found_images=sum(x['found_at']is not None for x in camera_files.values()))),flush=True)


if __name__=='__main__':main()
