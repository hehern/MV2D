#根据token，经过sample.json获取对应的scene_token，根据场景token判断是哪个地图

"""
    pkl组织:
    1.将车道线导出到json中;
    2.对车道线进行插值，获取密集车道线点;
    3.删除sweeps;
    4.删除cams字段中除前向之外的其他图片;
    5.根据token经过sample.json查找scene_token,再经过scene.json查找log_token,再根据map.json判断位于哪个地图上。
    
"""
from tqdm import tqdm
import numpy as np
import pickle
import yaml
import os
import json
import mmcv
import sys
import torch
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from scipy.interpolate import interp1d  
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes
# from tools.data_converter.pick_data import get_pointnums

dataroot = './data/nuscenes'
info_prefix = 'train'
# info_prefix = 'val'

# nuscenes = NuScenes('v1.0-trainval', dataroot)#用nuscenes.get的方法加载数据太慢了，舍弃
with open(dataroot+'/v1.0-trainval/sample.json', 'r', encoding='utf-8') as file:  
    sample_json = json.load(file) 
with open(dataroot+'/v1.0-trainval/scene.json', 'r', encoding='utf-8') as file:  
    scene_json = json.load(file) 
with open(dataroot+'/v1.0-trainval/map.json', 'r', encoding='utf-8') as file:  
    map_json = json.load(file)

map_tokens = {
    "36092f0b03a857c6a3403e25b4b7aab3": 'boston-seaport',
    "37819e65e09e5547b8a3ceaefba56bb2": 'singapore-hollandvillage',
    "53992ee3023e5494b90c316c183be829": 'singapore-onenorth',
    "93406b464a165eaba6d9de76ca09f5da": 'singapore-queenstown'
    }

map_locations = {
    'singapore-onenorth': list(),
    'singapore-queenstown': list(),
    'singapore-hollandvillage': list(),
    'boston-seaport': list()
    }

def draw_2d_points_to_image(img, points_2d, sample_token):
    draw = ImageDraw.Draw(img)  
    point_color = (255, 0, 0)
    
    for (x, y) in points_2d:
        draw.ellipse((x-3, y-3, x+3, y+3), fill=point_color)
    
    save_path = 'viz/lane/' + sample_token + '_lane.jpg'  # 替换为你想要保存的路径和文件名  
    img.save(save_path) 

def filter_lane_by_dis(map_lane, info, lane_ego_dis):
    nearby_map_lane = []
    ego_pose = torch.tensor([info["ego2global_translation"][0], info["ego2global_translation"][1]])
    for polygon in map_lane:
        global_cor = torch.tensor(polygon["points"])
        distances = torch.norm(global_cor - ego_pose, dim=1)
        if (distances < lane_ego_dis).any().item():
            nearby_map_lane.append(global_cor)
    return torch.cat(nearby_map_lane, dim=0)

def project_lane_to_img(img, map_lane, info, sample_token):

    # global to ego transform
    global2ego_r = np.linalg.inv(Quaternion(info["ego2global_rotation"]).rotation_matrix)
    global2ego_t = (
        info["ego2global_translation"] @ global2ego_r.T
    )
    global2ego_rt = np.eye(4).astype(np.float32)
    global2ego_rt[:3, :3] = global2ego_r.T
    global2ego_rt[3, :3] = -global2ego_t
    info["global2ego"] = global2ego_rt.T

    # ego to camera transform
    ego2camera_r = np.linalg.inv(Quaternion(info["sensor2ego_rotation"]).rotation_matrix)
    ego2camera_t = (
        info["sensor2ego_translation"] @ ego2camera_r.T
    )
    ego2camera_rt = np.eye(4).astype(np.float32)
    ego2camera_rt[:3, :3] = ego2camera_r.T
    ego2camera_rt[3, :3] = -ego2camera_t
    info["ego2camera"] = ego2camera_rt.T

    # camera intrinsics
    camera_intrinsics = np.eye(4).astype(np.float32)
    camera_intrinsics[:3, :3] = info["cam_intrinsic"]

    # global to image transform
    global2image = camera_intrinsics @ ego2camera_rt.T @ global2ego_rt.T
    info["global2image"] = global2image
    global2image_tensor = torch.tensor(global2image)

    # 筛选自车周围100m内的map_lane
    lane_ego_dis = 100.0
    global_cor = filter_lane_by_dis(map_lane, info, lane_ego_dis)

    # project
    points_2d = []
    width, height = img.size

    zero_tensor = torch.zeros(global_cor.shape[0], 1)
    global_cor_3d = torch.cat((global_cor, zero_tensor), dim=1).transpose(1, 0)#(3,n)
    cur_coords = global2image_tensor[:3, :3].matmul(global_cor_3d)
    cur_coords += global2image_tensor[:3, 3].reshape(3, 1)# torch.Size([1, 3, n]),将点云转换到图片坐标系

    # get 2d coords
    dist = cur_coords[2, :]
    cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
    cur_coords[:2, :] /= cur_coords[2:3, :]

    draw_points_2d = cur_coords[:2, :].transpose(1, 0)#(n,2)
    depth = cur_coords[2:3, :].transpose(1, 0)

    draw_points_2d[np.isnan(draw_points_2d)] = 0
    del_xindx, del_yindx = np.where(draw_points_2d[:, :] == 0)
    draw_points_2d = np.delete(draw_points_2d, del_xindx, axis=0)
    depth = np.delete(depth, del_xindx, axis=0)
    global_cor = np.delete(global_cor, del_xindx, axis=0)
    on_img = (
        (draw_points_2d[..., 0] < width)
        & (draw_points_2d[..., 0] >= 0)
        & (draw_points_2d[..., 1] < height)
        & (draw_points_2d[..., 1] >= 0)
    )

    depth_mask = ((depth > 0) & (depth < lane_ego_dis)).squeeze()
    draw_points_2d = draw_points_2d[on_img & depth_mask]
    global_cor = global_cor[on_img & depth_mask]

    if len(draw_points_2d) != 0:
        draw_2d_points_to_image(img, draw_points_2d, sample_token)
    else:
        print("no lane on this img: ", sample_token)
    # import ipdb; ipdb.set_trace()
    return (global_cor, draw_points_2d)

def filter_gt_by_cam_front(info):
    #info['gt_boxes']lidar坐标系，x_size, y_size, z_size, l, w, h, yaw
    #字段转换为8个角点坐标，依次投影到前向图片上,只要有一个角点在图片上就保留该标注
    gt_bboxes_3d = info["gt_boxes"]
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
    ).convert_to(Box3DMode.LIDAR)

def filter_pkl(info):
    # 1.删除无效字段
    del info['sweeps']
    del info['cams']['CAM_FRONT_RIGHT']
    del info['cams']['CAM_FRONT_LEFT']
    del info['cams']['CAM_BACK']
    del info['cams']['CAM_BACK_LEFT']
    del info['cams']['CAM_BACK_RIGHT']
    del info['gt_velocity']
    del info['num_radar_pts']

    # 2.根据当前sample token判断属于哪个地图
    sample_token = info['token']
    scene_token = next((entry['scene_token'] for entry in sample_json if entry['token'] == sample_token))  
    log_token = next((entry['log_token'] for entry in scene_json if entry['token'] == scene_token))
    map_token = next((entry['token'] for entry in map_json if log_token in entry['log_tokens']))
    map_name = map_tokens[map_token]
    map_lane = map_locations[map_name]

    # 3.将车道线点投影到前向图片上
    img = Image.open(info['cams']['CAM_FRONT']['data_path'])
    # img.save('viz/lane/'+sample_token+'.jpg')
    lane_3d, lane_2d = project_lane_to_img(img, map_lane, info['cams']['CAM_FRONT'], sample_token)

    # 4.将能投影到当前图片上的车道线点保存在lane字段中
    info['lane_3d'] = lane_3d #global坐标系下，需要再次转换到ego坐标系下
    info['lane_2d'] = lane_2d #前向图片坐标系下

    # 5.标注框过滤：只保留前向图片上的标注
    import ipdb; ipdb.set_trace()
    filter_gt_by_cam_front(info)

    # 6.添加2D标注框

    # info['num_lidar_pts'] = info['num_lidar_pts'][valid_mask]
    # info['gt_names'] = np.array(gt_names, dtype=str)[valid_mask]

    return True

def dense_lane(points):

    original_points = np.array(points)
    
    distance_m = 0.3
    new_points = list()
    for i in range(len(original_points)):  
        start_point = original_points[i]
        end_point = original_points[(i + 1) % len(original_points)]
        
        total_distance = np.sqrt((start_point[0]-end_point[0])**2 + (start_point[1]-end_point[1])** 2)
        num_points = int(total_distance / distance_m) - 1
        if num_points < 0:  
            new_points += [start_point, end_point] 
            continue
        interpolated_points = []  
        for i in range(1, num_points + 1):  
            t = i / (num_points + 1) 
            x = start_point[0] * (1 - t) + end_point[0] * t
            y = start_point[1] * (1 - t) + end_point[1] * t
            interpolated_points.append((x, y)) 
        new_points += [start_point] + interpolated_points + [end_point]

    new_points = np.vstack(new_points)
    
    # 可视化结果  
    # plt.figure()  
    # plt.plot(original_points[:, 0], original_points[:, 1], 'b-', linewidth=2)  # 使用蓝色实线连接点
    # plt.scatter(original_points[:, 0], original_points[:, 1], c='r', marker='o', label='Points')
    # plt.scatter(new_points[:, 0], new_points[:, 1], c='g', marker='o', label='Points')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Polygon Visualization')  
    # plt.xlabel('X')  
    # plt.ylabel('Y') 
    # plt.savefig('./interpolation_visualization.png')
    # import ipdb; ipdb.set_trace()

    return new_points.tolist()

def export_lane_json(map_name):

    # Open the NuScenes dataset.
    numap = NuScenesMap(dataroot=dataroot, map_name=map_name)
    lanes_data = list()

    for record in getattr(numap, 'lane'):#遍历所有的lane
        data = dict()
        data["token"] = record["token"]
        data["points"] = list()
        for token in record["exterior_node_tokens"]:   
            data["points"].append([numap.get('node', token)['x'], numap.get('node', token)['y']])
        data["points"] = dense_lane(data["points"])
        lanes_data.append(data)
    
    # save
    # with open(map_name+"_lane.json", 'w', encoding='utf-8') as json_file:  
    #     json.dump(lanes_data, json_file, ensure_ascii=False, indent=4)
    
    return lanes_data

if __name__ == "__main__":

    # step1: Loop through and export each map.
    for map_name in map_locations.keys():#遍历4个场景地图
        map_locations[map_name] = export_lane_json(map_name)#只导出车道线

    # step2: filter pkl
    key_infos = pickle.load(open(os.path.join(dataroot, 'nuscenes_infos_{}.pkl'.format(info_prefix)), 'rb'))
    info_path = os.path.join(dataroot, 'mmdet3d_nuscenes_front_lane_infos_{}.pkl'.format(info_prefix))

    data_infos = key_infos["infos"]
    filtered_data_infos = []

    for index in tqdm(range(0, len(data_infos))):
    # for index in range(0, len(data_infos)):
        info = data_infos[index]
        # 删除无效字段 & 投影车道线点并添加lane字段
        if filter_pkl(info):
            filtered_data_infos.append(info)

    # step3: save pkl
    data = dict(infos=filtered_data_infos)
    mmcv.dump(data, info_path)