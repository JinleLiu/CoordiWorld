"""R13/B23 scope and actual six-hour budget; old artifacts remain immutable."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime as dt
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));B22=ROOT/'twentysecond_20261028';BASE=ROOT/'twentythird_20261029';B21=ROOT/'twentyfirst_20261028';B20=ROOT/'twentieth_20261028';B19=ROOT/'nineteenth_20261028';B18=ROOT/'eighteenth_20261027';B17=ROOT/'seventeenth_20261027';B16=ROOT/'sixteenth_20261027'
PIN='0a380a9063d7162ec93d0f51e9990ebac585f720'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def check_time(reserve=3600):assert dt.datetime.now(dt.timezone.utc).timestamp()<dt.datetime.fromisoformat(json.load(open(BASE/'START.json'))['deadline_utc']).timestamp()-reserve
def record(run_id,hypothesis,change,data,result=None):
    write(BASE/'registry'/(run_id+'.json'),dict(run_id=run_id,created_utc=now(),hypothesis=hypothesis,change=change,data=data,result=result))
