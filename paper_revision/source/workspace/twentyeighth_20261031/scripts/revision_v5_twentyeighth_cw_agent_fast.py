"""Same v02 CW decisions/control; memoize identical immutable box polygons."""
import json
from revision_v5_twentyeighth_cw_agent import CoordiWorldAgent
from revision_v5_twentyeighth_polygon_cache import install
def get_entry_point():return 'CoordiWorldFastAgent'
class CoordiWorldFastAgent(CoordiWorldAgent):
 def setup(self,path):
  super().setup(path);self.polygon_memo=install();p=self.out/'agent_metadata.json';r=json.load(open(p));r['computation_only']='Exact immutable polygon cache; six float64 bit-pattern key, 8192 entries; same geometry algorithm/input and decisions';p.write_text(json.dumps(r,indent=2)+'\n')
 def destroy(self):
  if hasattr(self,'polygon_memo'):(self.out/'polygon_cache.json').write_text(json.dumps(self.polygon_memo.exact_cache_info()._asdict(),indent=2)+'\n')
  super().destroy()
