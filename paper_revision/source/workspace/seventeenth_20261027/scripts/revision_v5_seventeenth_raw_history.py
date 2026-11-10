import numpy as np
def raw_history_from_scene(scene):
    count=scene.scene_metadata.num_history_frames
    result=[]
    # Deliberately index only the past/current range; never inspect future frames.
    for i in range(count):
        frame=scene.frames[i];ego=frame.ego_status;pose=ego.ego_pose
        if not ego.in_global_frame:raise ValueError('NAVSIM scene ego pose must be global')
        annotations=frame.annotations
        result.append({'token':frame.token,'scene_token':scene.scene_metadata.scene_token,'log_name':scene.scene_metadata.log_name,'timestamp':int(frame.timestamp),'map_location':scene.scene_metadata.map_name,'ego2global_translation':[float(pose[0]),float(pose[1]),0.],'ego2global_rotation':[float(np.cos(pose[2]/2)),0.,0.,float(np.sin(pose[2]/2))],'ego_dynamic_state':np.r_[ego.ego_velocity,ego.ego_acceleration],'roadblock_ids':frame.roadblock_ids,'traffic_lights':frame.traffic_lights,'driving_command':ego.driving_command,'lidar_path':'.','cams':{name:{} for name in ['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0']},'anns':{'gt_boxes':annotations.boxes,'gt_names':annotations.names,'gt_velocity_3d':annotations.velocity_3d,'instance_tokens':annotations.instance_tokens,'track_tokens':annotations.track_tokens}})
    return result
