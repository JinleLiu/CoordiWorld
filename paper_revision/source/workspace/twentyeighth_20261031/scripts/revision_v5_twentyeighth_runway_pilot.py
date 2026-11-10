"""Complete paired engineering check after the already registered blocked pilot."""
import copy,datetime,hashlib,json,os,subprocess,time,xml.etree.ElementTree as E
from pathlib import Path
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');Q=B/'runway_pilot_run/bench2drive';Q.mkdir(parents=True,exist_ok=True);P=str(B.parent/'envs/v5/bin/python')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(Q/'REGISTRATION.json').exists()
proto=json.load(open(B/'bench2drive/PROTOCOL.json'));root=E.Element('routes');root.append(copy.deepcopy(next(r for r in E.parse(proto['xml']).getroot()if r.get('id')=='25378')));xml=Q/'routes.xml';E.ElementTree(root).write(xml,encoding='utf-8',xml_declaration=True);proto.update(cohort='Dev25378_shared_reference_runway',xml=str(xml),xml_sha=sha(xml),routes=[r.attrib for r in root]);methods=[]
template=json.load(open(B/'mapsupply_pilot_run/bench2drive/methods.json'))[0]
for variant in ['BaseEC','Full']:
    m=copy.deepcopy(template);m.update(id='CoordiWorld_'+variant+'_Runway_v05',agent=str(B/'scripts/revision_v5_twentyeighth_cw_agent_runway.py'));m['config'].update(method_id=m['id'],variant=variant,map_supply=False,policy_version='Original_'+variant+'_with_common_runway',bridge_version='v05_common_reference_runway');m['config']['implementation_sha']['revision_v5_twentyeighth_cw_agent_runway.py']=sha(m['agent']);m['config_sha']=hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest();methods.append(m)
write(Q/'PROTOCOL.json',proto);write(Q/'methods.json',methods);write(Q/'queue_control.json',dict(enabled_seeds=[17]));write(Q/'REGISTRATION.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Actual paired validation of the common truncated-reference runway repair; the small tail gain does not stand in for spatial-supply progress',variants=['BaseEC','Full'],scope='Exposed25378/seed17; comparison to frozen v02/v03/v04 engineering records only, separate cohort',no_supply=True,no_new_guard=True))
while len(list((B/'mapsupply_blocks_run/bench2drive/episodes').glob('seed17/*/*/FINAL.json')))<2:time.sleep(20)
for m in methods:
    env=os.environ.copy();env.update(R18_RUN_ROOT=str(Q),R18_ROUTE_XML=str(xml),R18_XML_SHA=proto['xml_sha'],R18_COHORT=proto['cohort'],R18_ONLY_METHOD=m['id'],R18_NATIVE_DEV_COMPLETE='1',R18_SINGLE_JOB='1',R18_QUEUE_WORKER='runway_pair',R18_RPC_PORT='29518',R18_TM_PORT='49518')
    with open(Q/(m['id']+'.log'),'a')as f:rc=subprocess.call([P,str(B/'scripts/queue_v09.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
write(Q/'SCHEDULE_STATUS.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='completed paired runway engineering check'))
