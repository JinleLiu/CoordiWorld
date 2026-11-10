"""Scientific figure from measured exposed-route execution-prefix errors."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
def run(raw,out):
 out.mkdir(parents=True,exist_ok=True);d=json.load(open(raw/'RESULT.json'));rows=list(csv.DictReader(open(raw/'per_route.csv')));moving=[r for r in rows if r['scope']=='moving_initial_speed_gt0p5'];routes=sorted({r['route']for r in moving},key=int);methods=['constant_velocity','pacifica','garage_pid'];labels=['Constant velocity','Pacifica + LQR','W4 PID + Garage'];colors=['#969696','#b15d45','#287982'];vals={m:np.array([float(next(r['mean_position_error_m']for r in moving if r['method']==m and r['route']==route))for route in routes])for m in methods};delta=vals['garage_pid']-vals['pacifica'];rng=np.random.default_rng(17);ci=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1),[.025,.975])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,ax=plt.subplots(1,2,figsize=(10.2,4.4),gridspec_kw={'width_ratios':[1.25,1]})
 y=np.arange(len(routes));ax[0].hlines(y,vals['garage_pid'],vals['pacifica'],color='#c3c3c3',lw=1.3)
 for m,label,col in zip(methods[1:],labels[1:],colors[1:]):ax[0].scatter(vals[m],y,color=col,s=27,label=label,zorder=3)
 ax[0].set_yticks(y,routes);ax[0].set_xlabel('Mean rear-position error at 0.5 s (m)');ax[0].set_ylabel('Exposed Dev10 route');ax[0].set_title('(a) Moving prefixes: paired route means',loc='left',fontsize=10);ax[0].grid(axis='x',alpha=.18)
 scopes=['all','moving_initial_speed_gt0p5','stationary'];x=np.arange(3);width=.24
 for j,(m,label,col)in enumerate(zip(methods,labels,colors)):
  vv=[d['summary'][s]['methods'][m]['route_equal_mean_position_error_m']for s in scopes];ax[1].bar(x+(j-1)*width,vv,width,color=col,label=label)
 ax[1].set_xticks(x,['All','Speed > 0.5 m/s','Speed ≤ 0.5 m/s']);ax[1].set_ylabel('Route-equal mean error (m)');ax[1].set_title('(b) Stationary / moving distinction',loc='left',fontsize=10);ax[1].grid(axis='y',alpha=.18);ax[1].legend(frameon=False,fontsize=8)
 fig.suptitle('Execution-prefix diagnostic — frozen current inputs, no policy outcome inferred',fontsize=11,y=.98);fig.text(.5,.02,'10 exposed routes · 583 moving / 1,374 low-speed prefixes · official controller replay exact at 19,619 ticks',ha='center',fontsize=8);fig.tight_layout(rect=[0,.045,1,.94])
 for ext in ['pdf','svg','png']:fig.savefig(out/f'execution_prefix.{ext}',dpi=190,bbox_inches='tight')
 (out/'per_route.csv').write_bytes((raw/'per_route.csv').read_bytes());(out/'source_result.json').write_bytes((raw/'RESULT.json').read_bytes())
 r=dict(scope='Development-only execution proxy evidence; not H3 driving-score benefit',moving_mean_delta_m=float(delta.mean()),route_bootstrap95_delta_m=ci.tolist(),moving_route_equal_relative_reduction=float(1-vals['garage_pid'].mean()/vals['pacifica'].mean()),moving_positive_routes=int((delta<0).sum()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [raw/'RESULT.json',raw/'per_route.csv']});(out/'FIGURE_SOURCE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw,a.out)
