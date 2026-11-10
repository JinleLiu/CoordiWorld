"""Read TRAIN identities and raw-log metadata only; never inspect outcome labels."""
import datetime
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path('<CW_ARCHIVE_ROOT>/coordiworld-rebuild-20261021')
RUN = ROOT / 'A_R18-followup-v03-20261102T1447'
RISK = ROOT / 'A_R18-followup-v02-20261102T1107/inputs/risk.npz'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write(name, obj):
    p = RUN / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def main():
    start = time.monotonic()
    with np.load(RISK, allow_pickle=False) as z:
        ids = z['scene_ids'].astype(str)
        drives = z['drives'].astype(str)
        folds = z['fold'].astype(int)
        refreshed = z['refreshed'].astype(bool)
    assert len(ids) == len(set(ids)) == 1755
    assert len(set(drives)) == 40
    assert all(len(set(folds[drives == d])) == 1 for d in set(drives))
    identity_rows = [dict(token=t, drive=d, fold=int(f), old_risk_row=i,
                          old_240_regression=bool(refreshed[i]))
                     for i, (t, d, f) in enumerate(zip(ids, drives, folds))]
    write('configs/NATURAL_DOMAIN_IDENTITIES.json', dict(
        created_at=now(), source=str(RISK), source_sha256=hashlib.sha256(RISK.read_bytes()).hexdigest(),
        selection_uses_outcome_labels=False, identity_count=len(ids), drive_count=len(set(drives)),
        rows=identity_rows))
    wanted = {r['token']: r for r in identity_rows}
    found = {t: [] for t in wanted}
    logs = sorted((ROOT / 'data/recovery/raw/logs/openscene-v1.1/meta_datas/trainval').glob('*.pkl'))
    matched = [(p, next((d for d in sorted(set(drives)) if p.stem.startswith(d + '_')), None)) for p in logs]
    matched = [(p, d) for p, d in matched if d is not None]
    inventories = []
    for number, (p, drive) in enumerate(matched, 1):
        before = time.monotonic()
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        frames = pickle.loads(data)
        assert isinstance(frames, list)
        hits = []
        for i, f in enumerate(frames):
            token = str(f['token'])
            if token not in wanted:
                continue
            assert wanted[token]['drive'] == drive
            ts = int(f['timestamp'])
            current_four = i >= 3 and all(0 < int(frames[j]['timestamp']) - int(frames[j-1]['timestamp']) < 550000 for j in range(i-2, i+1))
            previous_four = i >= 4 and all(0 < int(frames[j]['timestamp']) - int(frames[j-1]['timestamp']) < 550000 for j in range(i-3, i+1))
            future_ten = i + 10 < len(frames)
            row = dict(raw_log=str(p), raw_log_sha256=digest, raw_log_bytes=len(data),
                       log=p.stem, current_index=i, timestamp_us=ts, raw_frames=len(frames),
                       current_four_available=bool(current_four),
                       predecessor_four_available=bool(previous_four),
                       future_ten_available=bool(future_ten))
            found[token].append(row)
            hits.append(token)
        inventories.append(dict(path=str(p), sha256=digest, bytes=len(data), drive=drive,
                                frames=len(frames), matched_tokens=hits,
                                seconds=time.monotonic()-before))
        del frames, data
        write('evidence/SOURCE_AUDIT_LIVE.json', dict(status='RUNNING', at=now(), pid=os.getpid(),
              logs_done=number, logs_total=len(matched), tokens_found=sum(bool(v) for v in found.values()),
              seconds=time.monotonic()-start))
    rows = []
    for token, base in wanted.items():
        options = found[token]
        if options:
            assert len({x['timestamp_us'] for x in options}) == 1
            options.sort(key=lambda x: (not x['predecessor_four_available'], not x['future_ten_available'], x['raw_log']))
            row = dict(base, **options[0], alternative_source_count=len(options))
        else:
            row = dict(base, raw_log=None, missing_reason='token_not_found_in_existing_trainval_logs')
        rows.append(row)
    rows.sort(key=lambda x: (x['drive'], x.get('timestamp_us', -1), x['token']))
    write('evidence/TRAIN_SOURCE_INDEX.json', dict(created_at=now(), source_identity_count=len(ids), rows=rows))
    write('evidence/TRAIN_RAW_LOG_INVENTORY.json', inventories)
    summary = dict(status='PASS', at=now(), pid=os.getpid(), source_scenes=len(ids), source_drives=len(set(drives)),
                   raw_log_count=len(matched), raw_log_bytes=sum(x['bytes'] for x in inventories),
                   matched_scenes=sum(x['raw_log'] is not None for x in rows),
                   current_four_available=sum(x.get('current_four_available', False) for x in rows),
                   predecessor_four_available=sum(x.get('predecessor_four_available', False) for x in rows),
                   future_ten_available=sum(x.get('future_ten_available', False) for x in rows),
                   metadata_only=True, new_native_labels=0, new_model_fits=0,
                   seconds=time.monotonic()-start)
    write('evidence/SOURCE_AUDIT_RECEIPT.json', summary)
    write('evidence/SOURCE_AUDIT_LIVE.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        write('evidence/SOURCE_AUDIT_FAILURE.json', dict(status='FAILED', at=now(), error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
