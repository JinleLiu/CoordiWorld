"""R11/B21 constants; no inherited deadlines, selectors or data expansion."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_k]='1'
import json,hashlib,datetime as dt
from pathlib import Path
ROOT=Path(__import__('coordiworld_paths').workspace_path(''))
BASE=ROOT/'twentyfirst_20261028'
B20=ROOT/'twentieth_20261028'
B19=ROOT/'nineteenth_20261028'
B18=ROOT/'eighteenth_20261027'
METHODS=['Geometry_HC','IC0','QR','QR_local']
PIN='0a380a9063d7162ec93d0f51e9990ebac585f720'
Q_SHA='84f71e15ba77a769e137a17749d0f7040d84204f1de5b258250218b20084e644'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def check_time(reserve=3600):
    end=dt.datetime.fromisoformat(json.load(open(BASE/'START.json'))['deadline_utc'])
    assert dt.datetime.now(dt.timezone.utc).timestamp()<end.timestamp()-reserve,'R11 delivery reserve reached'
def cohort(scope):
    if scope=='full140':return B20/'development_full',B18/'FULL_INPUT_LOCAL.json'
    if scope=='exposed408':return B19/'confirmation',B19/'CONFIRMATION_INPUT_v02.json'
    raise ValueError(scope)
