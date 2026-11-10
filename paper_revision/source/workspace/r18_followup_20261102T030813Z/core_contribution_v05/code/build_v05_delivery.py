"""Small evidence delivery after all frozen analysis. No simulation or archive of old raw history."""
from pathlib import Path
import json,hashlib,datetime,csv,zipfile,shutil,os,stat
from analyze_v05 import read,sha,write_csv

def build(v):
 pointer=v/'runtime/DELIVERY_POINTER.json'
 if pointer.exists():return read(pointer)
 a=read(v/'analysis/FINAL_v01.json');f=read(v/'protocol/CONFIRMATION_FREEZE_v01.json');bridge=read(v/'runtime/BRIDGE_ACCEPTANCE.json');assert a['completed']==a['scheduled']==f['episode_budget']and bridge['passed']
 stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M');out=v/'delivery'/('v05_'+stamp);out.mkdir(parents=True,exist_ok=False);prefix='CoordiWorld_另寻他路_核心贡献验证_v05_';names={k:prefix+n+'_'+stamp+ext for k,n,ext in [('stage','阶段报告','.txt'),('facts','论文可用事实','.txt'),('results','结果','.csv'),('handoff','交接_B-to-A','.txt'),('zip','主审轻量包','.zip')]}
 p=a['comparisons']['PRIMARY'];s=a['comparisons']['SECONDARY'];arms=a['arms'];start=read(v/'START_RECEIPT.json');d=p['DS_difference'];lo,hi=p['CI95'];positive=p['positive_routes']
 if d<=0:decision='未支持地图定向供给相对合理 Generic 的增量；不继续优化该供给分支。'
 elif lo<=0:decision='不采用稳定独有优势主张；仅保留局部正向机制线索。95%区间包含零，未达到+2 DS工作参考线，停止本供给分支的继续优化。'
 elif d<2:decision='存在正向统计信号，但未达到主审工作性的+2 DS参考线；实际价值受限。'
 else:decision='在本次固定运行范围内支持地图定向供给的增量；采用仍需结合逐路线分布及风险、进展和成本取舍。'
 secondary_decision='受限支持：H-Map相对原池有局部正向点估计，但区间包含零且集中于少数路线；不升级为稳定供给收益。' if s['CI95'][0]<=0 else '支持固定运行范围内的补充供给作用；需同时报告新增风险和成本。'
 risk=[]
 for e in ['collisions_vehicle','collisions_pedestrian','collisions_layout','red_light','stop_infraction','outside_route_lanes']:
  x=arms['H-Map']['events'][e];y=arms['H-Generic']['events'][e];risk.append(f"{e}: H-Map {x['all_scheduled_total']}，H-Generic {y['all_scheduled_total']}；已知事件分母 {x['known_episodes']}/{y['known_episodes']}")
 def mean(m,k):return arms[m][k]['route_equal_mean']
 def diag(m,k):return arms[m]['motion'][k]['route_equal_mean']
 table='\n'.join(f"{m}: DS {mean(m,'DS'):.6f}; RC {mean(m,'RC'):.6f}; 成功 {arms[m]['success_integer']}/{arms[m]['success_denominator']}; 官方记录 {arms[m]['official_records']}/{arms[m]['episodes']}; 官方状态 {arms[m]['status_counts']}; 官方超时/阻塞事件或wall超时标记 {arms[m]['timeout_or_stall_episodes']}（不含TickRuntime状态，不能据此称无停滞）; 停车时间比例 {diag(m,'stopped_time_fraction_speed_lt_0_2')}; 全尝试槽时 {arms[m]['all_attempt_wall_seconds']/3600:.3f}小时"for m in f['methods'])
 scientific=f"唯一主要差 DS(H-Map)−DS(H-Generic) = {d:.6f}，95% route-cluster bootstrap区间 [{lo:.6f}, {hi:.6f}]；逐route正/负/零 = {positive}/{p['negative_routes']}/{p['zero_routes']}。\n次要差 DS(H-Map)−DS(H-Base) = {s['DS_difference']:.6f}，95%区间 [{s['CI95'][0]:.6f}, {s['CI95'][1]:.6f}]。\n"
 boundary='''三臂使用旧 CARLA MAP 输入平台，共享可得地图/当前actor/导航真值信息，不是纯传感器感知系统比较。三臂固定同一旧 CW 评价器，不证明 CW 优于 Simple；没有 NoWitness 对照，不证明 witness 的独立普遍必要性。不同版本/群体的分数不相减。旧 Supply-v2 Map-Simple−Generic-Simple = −6.659444，97.5%CI[−13.499167,0.187500]；Map-CW−Map-Simple = −17.940530，97.5%CI[−31.290185,−5.222366]，均原样保留。旧 MapTight 对匹配 Base 的局部正结果独立保存，不替代本轮主要比较。\n'''
 scope=f"固定{len(f['routes'])}条路线×3seed×3臂={f['episode_budget']}，桥接6项独立披露，不合并估计。按元数据和固定哈希选取4/2/2分层，排除已知60条曝光路线；上游公开预训练曝光未知。没有未见ParkingExit声明。每route先平均seed，再route等权；分层配对route-cluster bootstrap 50000次，固定seed20261105，保留所有臂与seed同簇。+2 DS仅为主审工作参考，不是公认阈值。无显著差异不等于等价。\n"
 phase='本线：固定科学队列与分析完成，科学剩余0，无自动下一轮。论文整体：仍需主审整合A/B证据、收窄核心主张并完成正式稿、b1及回复信；本轮完成不代表所有审稿意见已解决。\n'
 text=f"核心贡献验证v05｜封版 {stamp} 北京时间\n\n采用判断：{decision}\n支持：已核实同旧平台的公平三臂执行与局部供给机制。受限支持：相对Base的局部正向结果。未支持：Map稳定优于合理Generic、达到+2DS实际增量、CW优于Simple、witness普遍必要及更安全。\n{phase}\n{scientific}{secondary_decision}\n收益与代价\n{table}\n"+'\n'.join(risk)+f"\n\nH-Map−H-Generic RC差：{mean('H-Map','RC')-mean('H-Generic','RC'):.6f}。DS改善不能自动称为更安全。运动指标包含停车比例、移动条件加速度/jerk及距离/时间暴露分母；全部官方事件、缺失和失败见结果CSV。\n\n协议与执行\n{scope}T0_v05={start['T0_v05_beijing']}，原T0={start['original_T0_beijing']}，均未重置。冻结SHA={a['freeze_sha256']}。所有终结attempt保留，无追加或成绩驱动重试。\n桥接只核对版本、候选/资格/执行与控制绑定，不据桥接DS调参。相同seed不保证逐帧环境相同：主比较及次比较共有 {a['input_audit']['initial_mismatch_pairs']} 个首四帧不完全一致对，详见审计，主要分母不剔除。\n\n{boundary}\nv04已完成36保存状态诊断作为附录复用，0个学习修正版episode；预测排序诊断不是新闭环成绩。Generic离线草案修正及全部来源SHA保留。\n\n剩余关键工作：主审判断论文主张与A线独立证据的组合；本线无预先批准的后续科学队列。大型原始记录保留原服务器，轻量包提供原始路径和SHA索引，同盘保存不等于异机备份。\n"
 review=read(v/'analysis/FINAL_REVIEW_v01.json')
 extra='''\n最终核对与解释限制\n72项确认全部获得官方记录，0项接口错误、0项wall超时，未重试。Failed - TickRuntime共33项（Base12/Map10/Generic11），来自原平台4000tick终止条件，失败时实际记录约199.95秒。该上限与官方blocked/route_timeout事件分开；保留失败及其部分DS/RC，不作为成功。成功整数是预登记的完成且无适用处罚口径；Completed原始状态数分别12/14/13，不等于成功数6/6/6。\n唯一非零主要route为25865（ParkedObstacleTwoWays）：三个seed的Map−Generic DS差0、0.99、12.5，route均值+4.496667；其余7route为0。有限样本的95%区间触及零，不证明等价，也不支持跨route的稳定增量。\n主要比较新增车辆碰撞发生于25865 seed42：H-Map完成路线但与vehicle.ford.crown相撞，Generic在运行上限停下且无该碰撞；这是进展与风险取舍，不称为更安全。Map相对Base额外车辆碰撞发生于25865 seed17和42。两条比较均没有凭事件少、低jerk或碰撞/km单项判胜。\n初始观测差异：主要比较3410 seed23、3749 seed23；次比较3410 seed23。主要非零route25865的三seed首四帧均匹配，但同seed/同初始输入不意味着后续逐帧相同。按预登记提供差异分层描述，主分母和CI不剔除任何单元。\n供给机制：Map有12660次槽位尝试、67准入、9次最终选中；Generic有13320尝试、53准入、11次最终选中。次数是跨重规划的重复尝试，不是独立场景或不同曲线数量。各决策最大预算一致，实际尝试不同；本次新路线无ParkingExit，慢速两槽未触发。多数拒绝来自current_stationary_box与曲率，未据此放松限制。\n导出修复：原冻结分析脚本完整保留；只将供给尝试slot列改名supply_slot以避免与计算槽位重名。统计/失败/源码冻结规则未改，0次新增仿真；修复前错误、差异和原环境回归验证随包。\n'''
 motion_cost='\n移动条件指标（每episode p95再route等权，原始10Hz未平滑，不是合并样本p95；v>=0.2m/s）：\n'+'\n'.join(f"{m}: 纵向|jerk| {diag(m,'moving_lon_jerk_abs_p95'):.6f} m/s^3，横向|jerk| {diag(m,'moving_lat_jerk_abs_p95'):.6f} m/s^3；实际路径合计 {arms[m]['events']['collisions_vehicle']['actual_km']:.6f} km"for m in f['methods'])+'\n真实规划/控制调用计时和槽时见analysis/ACTUAL_COST_TABLE.csv，时钟包含当前三槽并行竞争，不声称独占部署实时性能。\n'
 text+=extra+motion_cost
 (out/names['stage']).write_text(text)
 (out/names['facts']).write_text('论文可用事实｜'+stamp+'\n\nRECALCULATED：'+scientific+scope+'\nMEASURED（官方字段及实测日志汇总）：'+table+'\n\nMEASURED：所有原始官方记录、attempt、运行时控制和计算成本均按固定分母导出。\nRECALCULATED：配对route均值与分层cluster区间来自本轮全矩阵，未合并桥接与历史分数。\nHISTORICAL：'+boundary+'\nUNVERIFIED：公开上游预训练曝光；现实道路安全泛化；CW相对Simple的正收益；witness独立普遍必要性。\n'+phase+extra+motion_cost)
 (out/names['handoff']).write_text('B-to-A 技术交接｜'+stamp+'\n\n'+decision+'\n'+scientific+'\n'+secondary_decision+'\n'+boundary+extra+motion_cost+'\nA的新头没有接入B，本轮也不外推A源域结果。共享区未删除，原模型与运行依赖保留。\n请主审用本包同平台比较收窄供给主张；不将实验完成等同返修全部完成。保存状态诊断只作机制说明。正式稿/b1/回复由主审写。\n完整索引：runtime/DELIVERY_POINTER.json；原始记录路径与SHA：RAW_EVIDENCE_INDEX.jsonl；全部来源/路线/协议和负结果随轻量包。未声称对方已采纳。\n')
 shutil.copyfile(v/'analysis/RESULTS_v01.csv',out/names['results'])
 claims=[dict(claim='Map-directed versus reasonable Generic',status=decision,evidence='PRIMARY comparison and per-route CSV',limit='fixed platform/8 routes; costs and uncertainty apply'),dict(claim='supplementary supply versus original pool',status=secondary_decision,evidence='SECONDARY comparison',limit='does not establish unique directional superiority'),dict(claim='CW readout beats Simple',status='NOT_SUPPORTED; prior negative preserved',evidence='sealed216',limit='v05 cannot answer this'),dict(claim='witness independently necessary',status='NOT_TESTED',evidence='no NoWitness arm',limit='historical cases only'),dict(claim='all reviewer concerns resolved',status='NOT_ESTABLISHED',evidence='chief manuscript integration pending',limit='experiment completion only')];write_csv(out/'CLAIM_EVIDENCE_MATRIX.csv',claims)
 inventory=out/'RAW_EVIDENCE_INDEX.jsonl';count=0;bytes_total=0
 with inventory.open('w')as target:
  for phase in ['bridge','confirmation']:
   q=v/(phase+'_v01')
   for pth in sorted(q.rglob('*')):
    if not pth.is_file()or pth.is_symlink():continue
    h=hashlib.sha256()
    with pth.open('rb')as src:
     for chunk in iter(lambda:src.read(4*1024*1024),b''):h.update(chunk)
    z=pth.stat();target.write(json.dumps(dict(phase=phase,path=str(pth),bytes=z.st_size,sha256=h.hexdigest()))+'\n');count+=1;bytes_total+=z.st_size
 (out/'STATE_AT_SEAL.json').write_text(json.dumps(dict(task='核心贡献验证v05',science_complete=True,scientific_remaining=0,original_T0=start['original_T0_beijing'],T0_v05=start['T0_v05_beijing'],freeze_sha256=a['freeze_sha256'],local_delivery_validation_pending=True),indent=2,ensure_ascii=False)+'\n')
 with(out/'OFFICIAL_RECORDS_v01.jsonl').open('w')as stream:
  for phase in ['bridge','confirmation']:
   for jobfile in sorted((v/(phase+'_v01')/'jobs').glob('*.json')):
    job=read(jobfile);raw=read(Path(job['final']));stream.write(json.dumps(dict(phase=phase,job=job['job'],official_record=raw['official_record'],source=job['final'],source_sha256=sha(Path(job['final'])),returncode=raw['returncode'],wall_timeout=raw['wall_timeout'],interface_errors=raw['parse_or_interface_errors']))+'\n')
 # Preserve first four frames and first selected-trajectory divergence, not full raw histories.
 from package_v05_windows import collect_windows
 window_files=collect_windows(v,a,out)
 # Include compact provenance/results, never raw full episode trees or former large delivery ZIPs.
 files={p.name:p for p in out.iterdir()if p.is_file()};files.update(window_files)
 for dirname in ['protocol','documentation','source','appendix_v04_saved_diagnostics','analysis']:
  for pth in sorted((v/dirname).rglob('*')):
   if pth.is_file()and '__pycache__'not in pth.parts:files[str(pth.relative_to(v))]=pth
 code_names=['h_base_agent.py','h_map_agent.py','h_generic_agent.py','h_generic_factory.py','h_generic_policy.py','h_generic_timing.py','h_observation.py','cw_controller_prefix.py','cw_followup_physics.py','owned_leaderboard_evaluator.py','v05_queue.py','validate_v05_episode.py','audit_v05_inputs.py','v05_statistics.py','v05_motion.py','analyze_v05.py','analyze_v05_export_fix_v01.py','test_v05_export_fix.py','summarize_v05_delivery.py','build_v05_delivery.py','package_v05_windows.py']
 for phase in ['bridge','confirmation']:
  files['runtime/'+phase.upper()+'_COORDINATOR_FINAL.json']=v/(phase+'_v01')/'COORDINATOR_FINAL.json'
 for name in code_names:files['code/'+name]=v/'code'/name
 for name in ['BRIDGE_ACCEPTANCE.json','BRIDGE_INPUT_AUDIT.json','CONFIRMATION_INPUT_AUDIT.json','ANALYSIS_VALIDATION_v01.json','BRIDGE_DISPATCH.json','CONFIRMATION_DISPATCH.json','BRIDGE_ACCEPTANCE_before_export_correction_v01.json','ANALYSIS_EXPORT_FIX_TEST_v01.json','FINAL_ANALYSIS.log','FINAL_ANALYSIS_EXPORT_FIX_v01.log','FINAL_ANALYSIS_RECEIPT.json','TICKRUNTIME_SOURCE_RECEIPT.json','BRIDGE_COORDINATOR_FINAL.json','CONFIRMATION_COORDINATOR_FINAL.json']:
  pth=v/'runtime'/name
  if pth.exists():files['runtime/'+name]=pth
 manifest={name:dict(bytes=pth.stat().st_size,sha256=sha(pth))for name,pth in files.items()};(out/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 archive=out/names['zip']
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,strict_timestamps=False)as z:
  for name,pth in files.items():z.write(pth,name)
  z.write(out/'PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.json')
 assert archive.stat().st_size<=50_000_000,'Light package exceeds50MB; no partial publication'
 with zipfile.ZipFile(archive)as z:
  assert z.testzip()is None
  for name,m in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
 result=dict(status='REMOTE_LIGHT_PACKAGE_VERIFIED',sealed_beijing=stamp,delivery_directory=str(out),files={k:dict(path=str(out/n),bytes=(out/n).stat().st_size,sha256=sha(out/n))for k,n in names.items()},raw_index_files=count,raw_index_bytes=bytes_total,zip_CRC_passed=True,all_payload_SHA_passed=True,zip_target_max_bytes=50_000_000,scientific_remaining=0,local_download_verified=False,old_T0_preserved=True)
 pointer.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');return result
if __name__=='__main__':
 import sys;print(json.dumps(build(Path(sys.argv[1]))))
