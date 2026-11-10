"""CPU-only recalculation of sealed saved metrics; no model/data/scorer imports.

Run from any directory: python recompute_saved_v05.py --output DIRECTORY
Uses only standard library and NumPy. Inputs are immutable numerical CSV exports.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['rawNC', 'NC', 'DAC', 'DDC', 'TLC', 'TTC', 'LK', 'HC', 'EC', 'EP']

def load(name):
    with (ROOT / 'data' / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')

def compare(left, right, name):
    l = {r['token']: r for r in left}
    r = {r['token']: r for r in right}
    assert len(l) == len(left) == 1336 and len(r) == len(right) == 1336
    assert set(l) == set(r)
    tokens = sorted(l)
    assert all(l[t]['drive'] == r[t]['drive'] for t in tokens)
    drives = sorted({l[t]['drive'] for t in tokens})
    assert len(drives) == 20
    ls = np.array([number(l[t]['Score']) for t in tokens])
    rs = np.array([number(r[t]['Score']) for t in tokens])
    assert np.isfinite(ls).all() and np.isfinite(rs).all(), 'Fixed denominator cannot shrink'
    delta = ls - rs
    sums, counts, means = [], [], []
    drive_rows = []
    for d in drives:
        mask = np.array([l[t]['drive'] == d for t in tokens])
        sums.append(delta[mask].sum()); counts.append(mask.sum()); means.append(delta[mask].mean())
        drive_rows.append(dict(comparison=name, drive=d, scenes=int(mask.sum()),
            left_Score=float(ls[mask].mean()), right_Score=float(rs[mask].mean()), delta_Score=float(delta[mask].mean())))
    sums, counts, means = np.array(sums), np.array(counts), np.array(means)
    ix = np.random.default_rng(170018).integers(len(drives), size=(10000, len(drives)))
    boot = sums[ix].sum(1) / counts[ix].sum(1)
    result = dict(comparison=name, scenes=len(tokens), drives=len(drives), scene_mean=float(delta.mean()),
        drive_mean=float(means.mean()), drive_cluster_CI95=np.quantile(boot,[.025,.975]).tolist(),
        positive_drives=int((means > 0).sum()), negative_drives=int((means < 0).sum()),
        zero_drives=int((means == 0).sum()), left_Score=float(ls.mean()), right_Score=float(rs.mean()), metrics={})
    events=[]
    for metric in METRICS:
        a=np.array([number(l[t][metric]) for t in tokens]); b=np.array([number(r[t][metric]) for t in tokens])
        ka=np.isfinite(a); kb=np.isfinite(b); known=ka & kb
        item=dict(paired_known=int(known.sum()), left_known=int(ka.sum()), right_known=int(kb.sum()),
            left_mean_known=float(a[ka].mean()) if ka.any() else None,
            right_mean_known=float(b[kb].mean()) if kb.any() else None)
        if metric != 'EP':
            added=known & (a<1) & (b>=1); repaired=known & (a>=1) & (b<1)
            item.update(left_events=int((a[ka]<1).sum()),right_events=int((b[kb]<1).sum()),
                added=int(added.sum()),repaired=int(repaired.sum()))
            for i in np.flatnonzero(added | repaired):
                events.append(dict(comparison=name, token=tokens[i], drive=l[tokens[i]]['drive'], metric=metric,
                    transition='added_by_left' if added[i] else 'repaired_by_left', left_value=float(a[i]),right_value=float(b[i])))
        result['metrics'][metric]=item
    return result, drive_rows, events

def assert_close(actual, expected, path='result'):
    if isinstance(expected, dict):
        for key in actual:
            if key in expected: assert_close(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        assert len(actual)==len(expected), path
        for i,(a,b) in enumerate(zip(actual,expected)): assert_close(a,b,path+f'[{i}]')
    elif isinstance(expected, (float,int)):
        assert np.isclose(actual, expected, rtol=0, atol=1e-10), (path,actual,expected)
    else:
        assert actual==expected,(path,actual,expected)

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    pins=json.loads((ROOT/'data/INPUT_SHA256.json').read_text())
    for name,digest in pins.items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest,name
    strategy=load('STRATEGY_PER_SCENE.csv'); conditional=load('FOUR_ARM_CONDITIONAL.csv')
    expected=json.loads((ROOT/'data/EXPECTED_COMPARISONS.json').read_text())
    jobs=[(strategy,'method','C1','S1','STRATEGY_C1_minus_S1')]
    jobs += [(conditional,'arm',a,b,'CONDITIONAL_'+a+'_minus_'+b) for a,b in [('C1','S1'),('C0','S0'),('C1','C0'),('S1','S0')]]
    results=[]; drives=[]; events=[]
    for table,key,a,b,name in jobs:
        v,d,e=compare([x for x in table if x[key]==a],[x for x in table if x[key]==b],name)
        assert_close(v,next(x for x in expected if x['comparison']==name));results.append(v);drives+=d;events+=e
    saved_events=load('EVENT_TRANSITIONS.csv')
    keys=('comparison','token','drive','metric','transition')
    assert sorted(tuple(x[k] for k in keys) for x in events)==sorted(tuple(x[k] for k in keys) for x in saved_events)
    assert len(events)==509
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'RECALCULATED_COMPARISONS.json').write_text(json.dumps(results,indent=2)+'\n')
    write_csv(args.output/'RECALCULATED_DRIVES.csv',drives);write_csv(args.output/'RECALCULATED_EVENTS.csv',events)
    receipt=dict(status='PASS',mode='saved-result CPU recalculation only',python=platform.python_version(),numpy=np.__version__,
        input_pins=pins,comparisons=5,paired_scenes_per_comparison=1336,drives=20,event_transition_rows=509,
        tolerance_absolute=1e-10,bootstrap_resamples=10000,bootstrap_seed=170018,
        new_fits=0,new_world_calls=0,new_native_calls=0,independent_validation=False)
    (args.output/'CPU_RECALCULATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__': main()
