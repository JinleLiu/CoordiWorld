"""v03 bridge: original frozen policy plus valid-short-rail runtime repair."""
import json,hashlib
from pathlib import Path
from revision_v5_twentyeighth_cw_agent_prepared import CoordiWorldPreparedAgent
import revision_v5_twentyeighth_short_path as shortpath
def get_entry_point():return 'CoordiWorldShortPathAgent'
class CoordiWorldShortPathAgent(CoordiWorldPreparedAgent):
 def setup(self,path):
  shortpath.install();super().setup(path)
  p=self.out/'agent_metadata.json';m=json.load(open(p));m['short_path_repair']=dict(version='v03',source_sha256=hashlib.sha256(Path(shortpath.__file__).read_bytes()).hexdigest(),operation='only recover upstream assertion for valid distinct map endpoints; no extrapolation; normal paths identical',proof='bridge/short_path_probe/result.json');p.write_text(json.dumps(m,indent=2)+'\n')
 def destroy(self):
  if hasattr(self,'out'):(self.out/'short_path_stats.json').write_text(json.dumps(shortpath.STATS,indent=2)+'\n')
  super().destroy()
