"""Actual-motion diagnostics, separate from learned/proxy EC; no filtering or imputation."""
import math
from confirmation_statistics import quantile
def summary(values):return dict(n=len(values),mean=sum(values)/len(values)if values else None,p50=quantile(values,.5),p95=quantile(values,.95))
def motion(traces,metric_rows,kind):
    speed_by_time={};speeds=[];xy=[];invalid=0;last=None
    for r in traces:
        try:
            t=float(r['sim_seconds']);v=[float(x)for x in r['velocity_world'][:2]]
            p=r.get('body_pose',r.get('pose'))
            if p is None:p=[r['ego_world']['x'],-r['ego_world']['y']]
            p=list(map(float,p[:2]));assert all(math.isfinite(x)for x in [t,*v,*p])and(last is None or t>last)
            last=t;s=math.hypot(*v);speeds.append(s);speed_by_time[round(t,6)]=s;xy.append((t,p))
        except (KeyError,ValueError,TypeError,AssertionError):invalid+=1
    # Do not bridge invalid trace gaps to manufacture a distance total.
    distance=sum(math.hypot(b[1][0]-a[1][0],b[1][1]-a[1][1])for a,b in zip(xy,xy[1:]))/1000 if len(xy)>1 and invalid==0 else None
    acceleration=[];metric_invalid=0;unmatched_speed=0;prior=None
    for r in metric_rows:
        try:
            t=float(r['sim_seconds']);a=r['acceleration'];forward=r['forward_vector'];right=r['right_vector'];assert all(len(v)==3 for v in [a,forward,right])
            assert all(math.isfinite(float(x))for v in [a,forward,right]for x in v)and math.isfinite(t)and(prior is None or t>prior);prior=t
            s=speed_by_time.get(round(t,6));unmatched_speed+=int(s is None)
            acceleration.append(dict(t=t,lon=sum(float(x)*float(y)for x,y in zip(a,forward)),lat=sum(float(x)*float(y)for x,y in zip(a,right)),speed=s))
        except (KeyError,ValueError,TypeError,AssertionError):metric_invalid+=1
    jerk=[];excluded_intervals=0
    for a,b in zip(acceleration,acceleration[1:]):
        dt=b['t']-a['t']
        if not .095<=dt<=.105:excluded_intervals+=1;continue
        jerk.append(dict(lon=(b['lon']-a['lon'])/dt,lat=(b['lat']-a['lat'])/dt,moving=a['speed']is not None and b['speed']is not None and a['speed']>.1 and b['speed']>.1))
    moving=[a for a in acceleration if a['speed']is not None and a['speed']>.1]
    allj=jerk;movj=[j for j in jerk if j['moving']]
    result=dict(trace_samples=len(traces),valid_trace_samples=len(speeds),invalid_trace_samples=invalid,actual_logged_ego_path_km=distance,stationary_sample_fraction_v_le_0_1=sum(v<=.1 for v in speeds)/len(speeds)if speeds else None,speed_mps=summary(speeds),moving_speed_mps=summary([v for v in speeds if v>.1]),metric_10Hz_samples=len(metric_rows),valid_metric_samples=len(acceleration),invalid_metric_samples=metric_invalid,metric_samples_without_exact_time_speed=unmatched_speed,moving_metric_samples=len(moving),valid_jerk_intervals=len(jerk),excluded_jerk_time_gap_intervals=excluded_intervals,moving_jerk_intervals=len(movj),moving_threshold_mps=.1,jerk_interval_seconds=[.095,.105],method_kind=kind,units=dict(acceleration='m/s^2',jerk='m/s^3'),definition='Raw world acceleration projected onto recorded vehicle forward/right unit axes; unfiltered adjacent10Hz component difference over observed dt. Moving jerk requires both endpoint speeds>0.1m/s. No counterfactual/proxy EC interpretation.')
    for axis in ['lon','lat']:
        result[axis+'_accel_abs']=summary([abs(a[axis])for a in acceleration]);result['moving_'+axis+'_accel_abs']=summary([abs(a[axis])for a in moving]);result[axis+'_jerk_abs']=summary([abs(j[axis])for j in allj]);result['moving_'+axis+'_jerk_abs']=summary([abs(j[axis])for j in movj])
    return result
def self_test():
    traces=[dict(sim_seconds=i*.05,velocity_world=[2,0,0],body_pose=[2*i*.05,0,0])for i in range(5)]
    metrics=[dict(sim_seconds=i*.1,acceleration=[i*.3,-i*.2,0],forward_vector=[1,0,0],right_vector=[0,1,0])for i in range(3)]
    x=motion(traces,metrics,'internal');assert abs(x['actual_logged_ego_path_km']-.0004)<1e-12 and abs(x['moving_lon_jerk_abs']['p95']-3)<1e-12 and abs(x['moving_lat_jerk_abs']['p95']-2)<1e-12
    assert x['moving_metric_samples']==3 and x['moving_jerk_intervals']==2
    metrics[-1]['sim_seconds']=.3;y=motion(traces,metrics,'internal');assert y['excluded_jerk_time_gap_intervals']==1 and y['metric_samples_without_exact_time_speed']==1
    assert motion(traces,[],'internal')['lon_accel_abs']['p95']is None
    return dict(passed=True,checks=['physical units and projections','20Hz path integration','unfiltered10Hz jerk','moving endpoint coverage','missing telemetry not imputed','gaps not smoothed'])
if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
