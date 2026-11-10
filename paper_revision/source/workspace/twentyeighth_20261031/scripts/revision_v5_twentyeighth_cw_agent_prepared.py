"""v02 policy with exactly equivalent prepared static map queries."""
import json,hashlib
from pathlib import Path
from revision_v5_twentyeighth_cw_agent_gc import CoordiWorldGCAgent
import revision_v5_twentyeighth_prepared_map as prepared
def get_entry_point():return 'CoordiWorldPreparedAgent'
class CoordiWorldPreparedAgent(CoordiWorldGCAgent):
 def setup(self,path):
  prepared.install(False);super().setup(path)
  p=self.out/'agent_metadata.json';m=json.load(open(p));m['prepared_map_queries']=dict(operation='prepare immutable geometry; contains(polygon,point) equals within(point,polygon); filter requested layer before exact predicate',source_sha256=hashlib.sha256(Path(prepared.__file__).read_bytes()).hexdigest(),proof='B28/bridge/prepared_probe: three real routes, nine contexts, full saved-array equality and per-call original-predicate verification',no_coordinate_rounding=True,no_metric_or_controller_changes=True);p.write_text(json.dumps(m,indent=2)+'\n')
 def destroy(self):
  if hasattr(self,'out'):(self.out/'prepared_map_stats.json').write_text(json.dumps(prepared.STATS,indent=2)+'\n')
  super().destroy()
