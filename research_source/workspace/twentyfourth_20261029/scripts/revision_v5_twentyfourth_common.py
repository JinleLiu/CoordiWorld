"""R14/B24 immutable run boundaries and explicit compatibility domain."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_k]='1'
from pathlib import Path
import json,hashlib,datetime as dt
ROOT=Path(__import__('coordiworld_paths').workspace_path(''))
BASE=ROOT/'twentyfourth_20261029'
B23=ROOT/'twentythird_20261029'
PIN='0a380a9063d7162ec93d0f51e9990ebac585f720'
DOMAIN='privileged_NAVSIM_IDM_rear_axle_corrected_endpoint_compat_v2'
METHODS=['PNC_H3P','PNC_NULL','PNC_EC','IC0_H3P','Geometry_H3']
POSE_VERSION='R14_official_pyquaternion_yaw_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def check_time(reserve=1800):
    end=dt.datetime.fromisoformat(json.load(open(BASE/'START.json'))['deadline_utc'])
    if dt.datetime.now(dt.timezone.utc).timestamp()>=end.timestamp()-reserve:
        raise TimeoutError('R14 actual deadline with delivery reserve reached')
