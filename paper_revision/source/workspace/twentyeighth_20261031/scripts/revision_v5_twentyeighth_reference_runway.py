"""Common bridge v05: pad a truncated reference along actual map successors.

Only current static connectivity and light status are used. A waypoint rail
ending at the provided route boundary is not an obstacle. The mission target
does not move; the real runner still decides completion at the original goal.
"""
import types
import numpy as np
CONFIG=dict(version='reference_runway_v02',max_padding_m=10.,max_heading_change_rad=np.pi/4,max_connection_gap_m=1.,min_successor_length_m=2.)

def install(policy):
    planner=policy.pipe.proposer.proposer.planner;original=planner._get_discrete_centerline;oldplan=policy.plan
    policy.reference_runway_records=[];policy.reference_runway_current=None
    def get_path(self,current_lane,search_depth=30):
        points=original(current_lane,search_depth);h=policy.reference_runway_current
        if h is None or not points:return points
        end=np.array([points[-1].x,points[-1].y,points[-1].heading]);lanes=list(self._route_lane_dict.values())
        matches=[l for l in lanes if np.linalg.norm(l.baseline_path.a[-1,:2]-end[:2])<1e-4]
        if not matches:return points
        tail=min(matches,key=lambda l:l.id);red={str(i)for i,r in h['traffic_lights']if r};choices=[]
        for nxt in tail.outgoing_edges:
            p=nxt.baseline_path.a;gap=float(np.linalg.norm(p[0,:2]-end[:2]));angle=float(abs(np.arctan2(np.sin(p[0,2]-end[2]),np.cos(p[0,2]-end[2]))))
            if gap>CONFIG['max_connection_gap_m']or angle>CONFIG['max_heading_change_rad']or nxt.baseline_path.length<CONFIG['min_successor_length_m']:continue
            arc=np.r_[0,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).cumsum()];deviation=float(abs(np.arctan2(np.sin(p[arc<=CONFIG['max_padding_m'],2]-end[2]),np.cos(p[arc<=CONFIG['max_padding_m'],2]-end[2]))).mean())
            choices.append((deviation,angle,gap,nxt.id,nxt))
        if not choices:return points
        choices.sort(key=lambda x:x[:4]);nxt=choices[0][-1]
        # A red intended continuation never licenses picking a different turn.
        if nxt.id in red:
            policy.reference_runway_records.append(dict(timestamp_us=int(h['timestamp']),tail_lane=tail.row['key'],successor_lane=nxt.row['key'],applied=False,reason='current_red_on_geometric_continuation'))
            return points
        extension=[];length=0.;prior=end[:2]
        for p in nxt.baseline_path.discrete_path:
            d=float(np.linalg.norm(p.array[:2]-prior))
            if length+d>CONFIG['max_padding_m']:break
            if d>1e-4:extension.append(p);length+=d;prior=p.array[:2]
        if not extension:return points
        goal=h.get('navigation_goal_math');record=dict(timestamp_us=int(h['timestamp']),applied=True,tail_lane=tail.row['key'],successor_lane=nxt.row['key'],old_endpoint=end.tolist(),new_endpoint=[extension[-1].x,extension[-1].y,extension[-1].heading],padding_m=length,goal=goal,old_endpoint_goal_distance_m=None if goal is None else float(np.linalg.norm(end[:2]-np.asarray(goal[:2],float))),current_obstacle_logic_unchanged=True,current_red_successor_excluded=True)
        policy.reference_runway_records.append(record)
        return points+extension
    planner._get_discrete_centerline=types.MethodType(get_path,planner)
    def plan(h):
        policy.reference_runway_current=h[-1];before=len(policy.reference_runway_records);result=oldplan(h)
        result[2]['reference_runway']=policy.reference_runway_records[before:]
        # Separate compact receipt avoids modifying the frozen plan log writer.
        import json
        with open(policy.out/'reference_runway.jsonl','a')as f:f.write(json.dumps(dict(replan=result[2]['replan'],records=policy.reference_runway_records[before:]))+'\n')
        return result
    policy.plan=plan;return policy
