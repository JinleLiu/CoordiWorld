#!/usr/bin/env python3
"""Public checks: model hashes and a synthetic six-head smoke test.

This entry point does not reproduce benchmark predictions, confidence intervals,
or selected-risk losses from individual records. Use the separately supplied
numerical supplement for those calculations.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from coordiworld_release.policy import evaluate

def readj(path):return json.loads((ROOT/path).read_text())
def model_checks():
    manifest=readj('config/model_identity.json');checked=[]
    for group,heads in manifest['models'].items():
        for head,record in heads.items():
            assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256'],group+'/'+head
            checked.append(group+'/'+head)
    world=ROOT/'models/world/reference_world.pt'
    if not world.exists() and (ROOT/'scripts/assemble_world_checkpoint.py').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/assemble_world_checkpoint.py')],check=True,cwd=ROOT)
    if not world.exists():raise FileNotFoundError('Assemble models/world/reference_world.pt using the released checkpoint parts.')
    assert hashlib.sha256(world.read_bytes()).hexdigest()==manifest['world_checkpoint_sha256']
    # Entirely synthetic zeros; no real feature records or empirical fixture.
    a={'x674':np.zeros((12,674),np.float32),'features198':np.zeros((12,198),np.float32),
       'V24':np.zeros((12,24),np.float32),'R8':np.zeros((12,8),np.float32),
       'C32':np.zeros((12,32),np.float32),'valid':np.ones(12,bool),'geo':0}
    pred=evaluate(a,ROOT/'models/PNC')
    assert all(np.isfinite(v).all()for v in pred['raw'].values())
    assert np.isfinite(pred['scores']).all() and 0<=pred['choice']<12
    return {'unchanged_model_hashes':checked+['world'],
            'synthetic_smoke':{'rows':12,'input':'All numerical features zero, all candidates valid, geometry reference 0.',
                               'six_head_outputs_first_row':{h:float(v[0])for h,v in pred['raw'].items()},
                               'selected_index':pred['choice'],
                               'scope':'Execution smoke test only; synthetic inputs are not a benchmark or evidence of prediction accuracy.'}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'public_checks.json')
    args=parser.parse_args()
    result={'model_checks':model_checks(),'passed':True,
            'scope':'Public code/weight identity and synthetic execution checks only. All scientific numerical replay is supplied separately in the accompanying numerical supplement.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'model_hashes':13,'synthetic_smoke_heads':6,'output':str(args.output)},indent=2))
if __name__=='__main__':main()
