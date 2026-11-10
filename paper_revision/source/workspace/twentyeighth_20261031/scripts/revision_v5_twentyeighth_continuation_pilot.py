"""Register one matched real parking-entry pilot, reuse each finished guard slot."""
import copy,datetime,hashlib,json,os,subprocess,time,xml.etree.ElementTree as E
from pathlib import Path
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');Q=B/'continuation_pilot_run/bench2drive';Q.mkdir(parents=True,exist_ok=True);P=str(B.parent/'envs/v5/bin/python')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(Q/'REGISTRATION.json').exists()
old=B/'mapsupply_pilot_run/bench2drive';proto=json.load(open(old/'PROTOCOL.json'));root=E.Element('routes');root.append(copy.deepcopy(next(r for r in E.parse(proto['xml']).getroot()if r.get('id')=='3514')));xml=Q/'routes.xml';E.ElementTree(root).write(xml,encoding='utf-8',xml_declaration=True);proto.update(cohort='Dev3514_shared_support_and_no_stop_witness',xml=str(xml),xml_sha=sha(xml),routes=[r.attrib for r in root]);methods=[]
for m in json.load(open(old/'methods.json')):
    m=copy.deepcopy(m);is_map=m['config']['map_supply'];mid='CoordiWorld_MapTimingWitness_v01'if is_map else'CoordiWorld_BaseSupportWitness_v01';m.update(id=mid,agent=str(B/'scripts/revision_v5_twentyeighth_cw_agent_support.py'));m['config'].update(method_id=mid,bridge_version='v06_below_road_current_support',policy_version='MapTiming_v01_with_shared_no_stop_witness'if is_map else'BaseEC_with_shared_no_stop_witness',map_timing=is_map,continuation_guard=True,current_support=True)
    files=['revision_v5_twentyeighth_observed_support.py','revision_v5_twentyeighth_continuation_guard.py','revision_v5_twentyeighth_cw_agent_support.py','revision_v5_twentyeighth_map_timing_policy.py'];m['config']['implementation_sha'].update({n:sha(B/'scripts'/n)for n in files});m['config_sha']=hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest();methods.append(m)
write(Q/'PROTOCOL.json',proto);write(Q/'methods.json',methods);write(Q/'queue_control.json',dict(enabled_seeds=[17]));write(Q/'REGISTRATION.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cohort=proto['cohort'],hypothesis='Keep a qualified merge whose next4s static continuation exists, without creating a new high-scoring stop option. Same current support and witness for both arms.',evidence_sha256=sha(B/'map_supply/continuation_probe_v02/RESULT.json'),evidence='Saved timing replan1: raw1->witness13; original1/2 have no static continuation, slow13 has one. Initial choice13 unchanged. Below-road actor correction removes1 atlate69 but does not by itself change that choice. No counterfactual CARLA result claimed.',comparison='Matched frozen BaseEC vs MapSupply04+conditional timing01, both use shared no-new-stop witness and below-road20m support. Same original NC/DAC and scores. No P0/training.',stop='Complete both registered episodes and retain all failures; no new34 used in development.'))
launched={};prior=B/'viability_pilot_run/bench2drive'
for m in methods:print(m['id'],m['config_sha'],flush=True)
while len(launched)<2:
    for m in methods:
        mid=m['id'];is_map=m['config']['map_supply'];oldmid='CoordiWorld_MapSupply_Guard_v01'if is_map else'CoordiWorld_BaseEC_Guard_v01';port=29718 if is_map else 29318;worker='guard_map_transfer_h3'if is_map else'guard_map_transfer_gspeed'
        if mid in launched or len(list((prior/'episodes').glob(f'seed17/*/{oldmid}/FINAL.json')))!=2:continue
        (prior/f'PAUSE_{worker}').write_text('Both registered stop-guard episodes completed; release to matched no-stop witness pilot\n');env=os.environ.copy();env.update(R18_RUN_ROOT=str(Q),R18_ROUTE_XML=str(xml),R18_XML_SHA=proto['xml_sha'],R18_COHORT=proto['cohort'],R18_ONLY_METHOD=mid,R18_NATIVE_DEV_COMPLETE='1',R18_SINGLE_JOB='1',R18_QUEUE_WORKER='continuation_map'if is_map else'continuation_base',R18_RPC_PORT=str(port),R18_TM_PORT=str(port+20000))
        with open(Q/(mid+'.log'),'a')as f:p=subprocess.Popen([P,str(B/'scripts/queue_v09.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        launched[mid]=dict(pid=p.pid,port=port,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(Q/'SCHEDULE_STATUS.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launched=launched,completed=len(list((Q/'episodes').glob('seed*/*/*/FINAL.json'))),expected=2));time.sleep(20)
