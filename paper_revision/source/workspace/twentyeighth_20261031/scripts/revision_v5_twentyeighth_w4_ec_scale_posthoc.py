"""Saved same-input W4 score arithmetic only; does not execute a new policy."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def run(source,out):
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(source.glob('*.npz')):
        a=np.load(p,allow_pickle=False)
        ids=a['EC_H3_ids'];before=a['EC_H3_route'];after=a['EC_H3_final']
        chosen=int(after.argmax());plain=int(before.argmax());adjustment=after-before
        assert np.all(np.minimum(abs(adjustment),abs(adjustment-np.log(.01)))<2e-6)
        rows.append(dict(context=p.stem,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pool_count=len(ids),choice=int(ids[chosen]),argmax_before_EC=int(ids[plain]),choice_is_extra=bool(ids[chosen]>=64),choice_changed_by_EC=bool(chosen!=plain),before_EC_score_gap=float(before[plain]-before[chosen]),EC_adjustment_advantage=float(adjustment[chosen]-adjustment[plain]),chosen_before_EC=float(before[chosen]),best_before_EC=float(before[plain]),chosen_after_EC=float(after[chosen]),best_before_EC_candidate_after_EC=float(after[plain])))
    extra=[r for r in rows if r['choice_is_extra']]
    s=dict(scope='Post-freeze arithmetic on the existing78 saved same-RGB/ego/pool contexts, not random-population inference or a new closed-loop arm. The before-EC choice is a one-step diagnostic using the same rescored augmented pool; its later history/outcomes have not been run.',contexts=len(rows),extra_choices=len(extra),extra_already_argmax_before_EC=sum(not r['choice_changed_by_EC']for r in extra),extra_overrides_original_argmax=sum(r['choice_changed_by_EC']and r['argmax_before_EC']<64 for r in extra),score_gap_min=float(min(r['before_EC_score_gap']for r in extra)),score_gap_median=float(np.median([r['before_EC_score_gap']for r in extra])),score_gap_max=float(max(r['before_EC_score_gap']for r in extra)),registered_lambda=1.,registered_score='route_score + log(0.01 + 0.99 * EC), when own history is available',binary_EC_score_advantage=float(-np.log(.01)),interpretation='In47 of53 extra selections, the registered EC term overturns the same-pool pre-EC preference for an original candidate. This explains a decision mechanism; it does not prove a smaller penalty is a successful controller, or that any unexecuted candidate is physically safer.',model_or_policy_changed=False)
    assert s['contexts']==78 and s['extra_choices']==53
    with open(out/'per_context.csv','w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'RESULT.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.source,a.out)
