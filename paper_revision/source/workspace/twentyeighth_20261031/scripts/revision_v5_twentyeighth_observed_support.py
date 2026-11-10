"""Conservative current-observation support for CARLA's 2D planner view.

Some live, active CARLA actors are far BELOW the current road surface.
Reject only boxes whose entire vertical extent is at least20m below BOTH
their nearest mapped road and ego. No future state or scenario identity.
This is a shared input adapter correction, not a learned-risk relaxation.
"""
import copy
import numpy as np
CONFIG=dict(version='below_road_support_v01',below_road_margin_m=20.,below_ego_margin_m=20.,horizontal_range_m=100.,unknown_road='retain',future_labels=False)

def filtered_frame(frame,cmap):
    import carla
    h=copy.deepcopy(frame);an=h['anns'];yaw=h['body_pose_math'][2];c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s],[s,c]]);origin=np.asarray(h['ego2global_translation'][:2]);ego_z=h['ego2global_translation'][2];keep=[];removed=[]
    for i,b in enumerate(an['gt_boxes']):
        top=b[2]+b[5]/2
        if top>=ego_z-CONFIG['below_ego_margin_m']:keep.append(i);continue
        xy=origin+R@np.asarray(b[:2]);wp=cmap.get_waypoint(carla.Location(x=float(xy[0]),y=float(-xy[1]),z=float(b[2])),True,carla.LaneType.Any)
        road_z=None if wp is None else float(wp.transform.location.z)
        if road_z is None or top>=road_z-CONFIG['below_road_margin_m']:keep.append(i);continue
        removed.append(dict(token=an['instance_tokens'][i],box_top_z=float(top),road_z=road_z,ego_z=float(ego_z),below_road_m=road_z-top,below_ego_m=ego_z-top,original_box=b))
    if removed:
        h['anns']={k:[v[i]for i in keep]for k,v in an.items()}
    h['current_support']=dict(config=CONFIG,input_count=len(an['gt_boxes']),retained_count=len(keep),removed=removed)
    return h
