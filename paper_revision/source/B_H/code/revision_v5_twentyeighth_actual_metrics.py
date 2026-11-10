"""Saved real ego telemetry: pinned-script literal and separately named SI diagnostics."""
import sys,json,csv,importlib.util,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
B=Path(sys.argv[1]);O=Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True)
SRC=Path(sys.argv[3])if len(sys.argv)>3 else Path('<CW_WORKSPACE>/twentyseventh_20261030/external/Bench2Drive-7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/efficiency_smoothness_benchmark.py')
spec=importlib.util.spec_from_file_location('pinned_b2d_aux',SRC);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[];missing=[]
for f in sorted((B/'bench2drive/episodes').glob('seed*/*/*/FINAL.json')):
 r=json.load(open(f));d=f.parent/f"attempt{r['attempt']}"/'trace';p=d/'actual_metric_10hz.jsonl';key=dict(method=r['method'],route=r['route']['id'],seed=r['seed'])
 if not p.exists():missing.append(key|dict(reason='not captured; no imputation'));continue
 a=[json.loads(z)for z in open(p)]
 if len(a)<3:missing.append(key|dict(reason='fewer than3 captured samples'));continue
 t=np.array([x['sim_seconds']for x in a]);dt=np.diff(t);np.testing.assert_allclose(dt,.1,atol=1e-6,rtol=0)
 fields=['acceleration','angular_velocity','forward_vector','right_vector','location','rotation'];arr={k:np.asarray([x[k]for x in a],float)for k in fields};assert all(np.isfinite(v).all()for v in arr.values())
 literal=float(mod.seg_compute_comfort_metric(**arr));w=min(7,len(a));w=w if w%2 else w-1
 def smooth(v,deriv=0):return savgol_filter(v,w,2,deriv=deriv,delta=.1,axis=0)
 lon=np.sum(arr['acceleration']*arr['forward_vector'],axis=1);lat=np.sum(arr['acceleration']*arr['right_vector'],axis=1);omega=np.deg2rad(arr['angular_velocity'][:,2])
 metrics={'long_accel_mps2':smooth(lon),'lat_accel_mps2':smooth(lat),'long_jerk_mps3':smooth(lon,1),'angular_velocity_z_radps':smooth(omega),'angular_acceleration_z_radps2':smooth(omega,1)}
 row=key|dict(samples=len(a),duration_s=float(t[-1]-t[0]),observed_horizontal_distance_km=float(np.linalg.norm(np.diff(arr['location'][:,:2],axis=0),axis=1).sum()/1000),upstream_literal_smoothness_fraction=literal,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),official_script_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),SI_scope='3D acceleration projected onto actual body axes; angular velocity converted from degrees/s; SG7/poly2 at10Hz; no thresholds or frame exclusions. Observed distance is the10Hz horizontal polyline during captured telemetry, not official route-completion distance.')
 for name,v in metrics.items():row.update({name+'_rms':float(np.sqrt(np.mean(v*v))),name+'_abs_p95':float(np.quantile(abs(v),.95)),name+'_abs_max':float(abs(v).max())})
 speeds={}
 for trace in d.rglob('trace.jsonl'):
  for line in open(trace):
   x=json.loads(line)
   if x.get('velocity_world')is not None:speeds[round(x['sim_seconds']*1e6)]=float(np.linalg.norm(x['velocity_world'][:2]))
 vv=[speeds[round(x*1e6)]for x in t if round(x*1e6)in speeds]
 row.update(speed_samples=len(vv),actual_horizontal_speed_max_mps=float(np.max(vv))if vv else None,actual_horizontal_speed_p95_mps=float(np.quantile(vv,.95))if vv else None,actual_horizontal_speed_mean_mps=float(np.mean(vv))if vv else None,stationary_fraction_v_le_0_1=float(np.mean(np.array(vv)<=.1))if vv else None);rows.append(row)
if rows:
 with open(O/'actual_metrics_per_episode.csv','w',newline='')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(O/'actual_metrics_scope.json').write_text(json.dumps(dict(rows=len(rows),missing=missing,upstream_literal='Unmodified pinned function on raw get_metric_info arrays; this is an implementation-specific output, not a validated SI physical comfort score',caveat='Upstream consumes raw degrees/s angular velocity and its yaw-acceleration smoothing has no derivative argument. SI diagnostics here are separate and do not replace official DS/success. Stopping can improve smoothness; inspect progress and stationary fraction jointly.',sampler='post-decision current ego states; first callback and every2 thereafter; no model/control input change',older_data='not imputed'),indent=2)+'\n');print('real metric episodes',len(rows),'missing',len(missing))
