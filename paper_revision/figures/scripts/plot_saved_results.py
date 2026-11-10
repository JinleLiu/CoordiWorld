#!/usr/bin/env python3
"""Deterministic figures from sealed saved tables; no inference, scoring or bootstrap.
Run: python3 scripts/plot_saved_results.py (from the asset folder), or from any cwd.
"""
from pathlib import Path
import csv, json, hashlib, platform, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
R=Path(__file__).resolve().parents[1]; D=R/'data'; O=R/'outputs'; O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.titlesize':10,'axes.linewidth':.65,
 'lines.linewidth':1.0,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','figure.facecolor':'white'})
def rows(name):
 with (D/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,stem):
 for ext in ('pdf','svg','png'):fig.savefig(O/(stem+'.'+ext),dpi=300)
 plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dr=rows('A_drive_effects_v07.csv'); mapping=rows('A_drive_mapping_v07.csv')
paired=rows('A_paired_effects_v07.csv'); primary=next(r for r in paired if r['comparison']=='STRATEGY_C1_minus_S1')
assert len(dr)==20 and len({r['drive'] for r in dr})==20 and len({r['display_id'] for r in dr})==20
assert {r['display_id'] for r in dr}=={f'D{i:02}' for i in range(1,21)}
assert all(any(m['display_id']==r['display_id'] and m['drive']==r['drive'] for m in mapping) for r in dr)
n=sum(int(r['scene_count']) for r in dr); assert n==int(primary['scenes'])==1336
pos=sum(float(r['C1_minus_S1'])>0 for r in dr); neg=sum(float(r['C1_minus_S1'])<0 for r in dr)
assert (pos,neg)==(19,1)
overall=float(primary['scene_mean']); weighted=math.fsum(int(r['scene_count'])*float(r['C1_minus_S1']) for r in dr)/n
assert math.isclose(weighted,overall,rel_tol=0,abs_tol=1e-12)
lo,hi=float(primary['CI95_low']),float(primary['CI95_high'])
ordered=sorted(dr,key=lambda r:(float(r['C1_minus_S1']),r['display_id']))
fig=plt.figure(figsize=(3.5,4.65)); ax=fig.add_axes([.17,.19,.69,.73])
ys=np.arange(20,0,-1)
ax.axvline(0,color='#696969',lw=.7,zorder=0)
for y,r in zip(ys,ordered):
 v=float(r['C1_minus_S1']); c='#396b9a' if v>=0 else '#86694a'
 ax.plot([0,v],[y,y],color=c,lw=.8);ax.plot(v,y,'o',color=c,ms=3.6)
 ax.text(1.10,y,r['scene_count'],transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=8.5)
ax.set_yticks(ys,[r['display_id'] for r in ordered]);ax.tick_params(axis='y',length=0,pad=4)
ax.set_ylim(-2.8,21.4);ax.set_xlim(-1.5,17.0);ax.set_xticks([0,5,10,15]);ax.grid(axis='x',lw=.35,color='#dddddd')
ax.hlines(-.2,*ax.get_xlim(),color='#999999',lw=.6)
ax.errorbar(overall,-1.5,xerr=[[overall-lo],[hi-overall]],fmt='D',color='#244d75',ms=4,capsize=2,lw=1.2)
ax.text(-.03,-1.5,'Overall',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.5)
ax.text(1.10,-1.5,str(n),transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=8.5)
ax.text(1.10,21.1,'n',transform=ax.get_yaxis_transform(),ha='center',va='center')
ax.set_xlabel('C1 − S1 Score (points)',labelpad=5)
for side in ('top','left','right'):ax.spines[side].set_visible(False)
fig.text(.17,.965,'Policy-history comparison',weight='bold',fontsize=10)
fig.text(.17,.060,f'Overall: {overall:+.2f}  [95% CI {lo:.2f}, {hi:.2f}]',fontsize=9)
fig.text(.17,.023,'Scene-weighted; individual drives are means.',fontsize=8.5)
save(fig,'CW_Fig3_drive_effects_v12')

controls=rows('B_CASE_CONTROL_TIMELINE_v07.csv'); outcomes=rows('B_CASE_OUTCOMES_v07.csv')
prov=json.loads((D/'B_CASE_WINDOW_PROVENANCE_v07.json').read_text())
assert {r['method'] for r in controls}=={'H-Map','H-Generic'}
assert {r['sim_timestamp_us'] for r in prov}=={17550000}
anchor=prov[0]['sim_timestamp_us']/1e6
assert all(r['route']=='R001' and r['seed']=='42' for r in outcomes)
styles={'H-Map':dict(color='#285f91',ls='-',marker='o',mfc='white',ms=3.3),
 'H-Generic':dict(color='#b6712e',ls='--',marker='x',ms=3.5)}
series={}; gaps={}
for method in styles:
 rs=[r for r in controls if r['method']==method]
 rs=sorted(rs,key=lambda r:float(r['sim_seconds']))
 assert len({r['step'] for r in rs})==len(rs)
 assert all(float(b['sim_seconds'])>float(a['sim_seconds']) for a,b in zip(rs,rs[1:]))
 gaps[method]=[]; out=[];prev=None
 for r in rs:
  if prev and int(r['step'])!=int(prev['step'])+1:
   out.append(None);gaps[method].append([int(prev['step']),int(r['step'])])
  out.append(r);prev=r
 series[method]=out
tmin=min(float(r['sim_seconds'])-anchor for r in controls);tmax=max(float(r['sim_seconds'])-anchor for r in controls)
# Exact observed range and exact samples: ticks label relative time only.
def panel(fig,rect,field,label,showlegend=False,showxlabel=True,annotate=False):
 ax=fig.add_axes(rect)
 for method,rs in series.items():
  ts=[np.nan if r is None else float(r['sim_seconds'])-anchor for r in rs]
  vs=[np.nan if r is None or not r[field].strip() else float(r[field]) for r in rs]
  ax.plot(ts,vs,label=method,**styles[method],lw=.95,mew=.75)
 ax.axvline(0,color='#777777',lw=.7,ls=':');ax.set_xlim(tmin-.015,tmax+.015)
 ax.set_xticks([-.5,-.25,0,.25,.5]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 if field in ('brake','throttle'):ax.set_ylim(-.06,1.06);ax.set_yticks([0,.5,1])
 else:ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 ax.grid(color='#e1e1e1',lw=.4);ax.spines[['top','right']].set_visible(False)
 ax.text(0,1.40 if showlegend else 1.06,label,transform=ax.transAxes,fontsize=9,weight='bold')
 if showxlabel:ax.set_xlabel('Time from first plan divergence (s)',fontsize=8.5,labelpad=3)
 if showlegend:ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8.5,frameon=False,ncol=2,handlelength=1.4,columnspacing=.7,borderaxespad=.1)
 if annotate:ax.annotate('First plan divergence',xy=(0,1),xycoords=('data','axes fraction'),xytext=(.04,1.22),textcoords='axes fraction',fontsize=8.5,arrowprops=dict(arrowstyle='-',lw=.6,color='#777777'))
 return ax
for i,(field,label) in enumerate([('steer','Steering (normalized)'),('brake','Brake'),('throttle','Throttle')]):
 f=plt.figure(figsize=(3.5,1.90 if i==0 else 1.55 if i==1 else 1.75));panel(f,[.19,.22 if i==0 else .25 if i==1 else .31,.72,.50 if i==0 else .56 if i==1 else .50],field,label,showlegend=i==0,showxlabel=i==2)
 save(f,'CW_controls_'+field+'_v12')
fig=plt.figure(figsize=(3.5,5.35))
for i,(field,label) in enumerate([('steer','Steering (normalized)'),('brake','Brake'),('throttle','Throttle')]):
 panel(fig,[.19,.70-i*.26,.72,.17],field,label,showlegend=i==0,showxlabel=i==2)
fig.text(.19,.975,'Route R001 · seed 42',weight='bold',fontsize=10)
fig.text(.03,.052,'Episode outcomes (separate records):',fontsize=8.5,weight='bold')
hm=next(r for r in outcomes if r['method']=='H-Map');hg=next(r for r in outcomes if r['method']=='H-Generic')
fig.text(.03,.030,f"H-Map: DS {float(hm['DS']):.2f}, Completed; 1 vehicle collision.",fontsize=8)
fig.text(.03,.009,f"H-Generic: DS {float(hg['DS']):.2f}, TickRuntime. Neither strict success.",fontsize=8)
save(fig,'CW_controls_combined_preview_v12')
checks={'drives':len(dr),'scenes':n,'positive_drives':pos,'negative_drives':neg,'weighted_drive_mean':weighted,'official_scene_mean':overall,'absolute_difference':abs(weighted-overall),'saved_CI':[lo,hi],'display_order':[r['display_id'] for r in ordered],
 'control_samples':{m:len([r for r in rs if r]) for m,rs in series.items()},'missing_step_gaps':gaps,'time_anchor_seconds':anchor,'relative_time_range':[tmin,tmax],'no_resampling':True,'no_new_confidence_intervals':True}
(R/'DATA_CHECKS_v12.json').write_text(json.dumps(checks,indent=2)+'\n')
sources=json.loads((R/'SOURCE_FILES.json').read_text());assert all(sha(R/r['copy'])==r['sha256'] for r in sources)
manifest={'version':'v12','sources':sources,'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
 'scripts':[{'path':'scripts/plot_saved_results.py','sha256':sha(Path(__file__))}],
 'figures':[
 {'name':'CW_Fig3_drive_effects_v12','fields':['display_id','drive','scene_count','C1_minus_S1','comparison','scene_mean','CI95_low','CI95_high'], 'filter':'All 20 drive rows; Overall uses STRATEGY_C1_minus_S1 saved row.', 'deterministic_transform':'Sort ascending full-precision C1_minus_S1; stable original display_id for ties. Compute weighted mean only as identity check. Reuse saved Overall CI; no per-drive CI.', 'display':'3.5 x 4.65 inches; 8.5–10 pt; single linear axis; original D mapping retained.'},
 {'name':'CW_controls_*_v12','fields':['method','step','sim_seconds','steer','brake','throttle','DS','RC','status','success','collisions_vehicle','sim_timestamp_us'], 'filter':'Entire retained route R001 seed42 display window, steps 340–360, each method independently.', 'deterministic_transform':'Sort each method by its own sim_seconds; subtract source anchor 17550000 us / 1e6. Keep exact sample timestamps and values. Insert NaN breaks for missing steps/values. No interpolation or vertical offset.', 'outcomes':'Separate sealed episode rows; divergence marker is not collision time.'}],
 'outputs':[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file()], 'checks':checks,
 'publication_status':'Internal candidate; third-party/data distribution scope requires publishing-owner review.'}
(R/'FIGURE_DATA_MANIFEST_v12.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(R/'README.md').write_text('''# CoordiWorld v12 deterministic paper figures

Run from this directory: `python3 scripts/plot_saved_results.py` (or invoke the script by absolute path).
Requires Python, NumPy and Matplotlib; exact versions and source/output SHA-256 are in FIGURE_DATA_MANIFEST_v12.json.
The data/ files are byte-identical copies of the retained v07 plotting inputs. No model, inference, simulation, scoring or bootstrap is invoked.
All changes are display transformations. The combined controls preview is a review convenience; LaTeX composes the three individual panels.
Source metadata contains internal provenance paths. This is an internal handoff; review distribution permissions and paths before public release.
Figures 1/2 and all scientific assets outside this folder are unchanged.
''')
print(json.dumps(checks,indent=2))
