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
from visualize import *

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

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]#xyz的顺序，目前lidar frame是右前天的顺序
point_cloud_range = [-20.0, 0.0, -5.0, 20.0, 60.0, 3.0]#xyz的顺序，目前lidar frame是右前天的顺序

def draw_2d_points_to_image(img, info):
    sample_token = info['token']

    # 前向图片绘制车道线投影
    draw = ImageDraw.Draw(img)  
    point_color = (255, 0, 0)
    
    for points_2d in info['lane_2d']:
        for (x, y) in points_2d:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=point_color)

    # 前向图片绘制3dbox投影
    image = np.array(img)
    bboxes = info['gt_boxes']
    bboxes[..., 2] -= bboxes[..., 5] / 2
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
    gt_labels_3d = []
    for cat in info['gt_names']:
        if cat in class_names:
            gt_labels_3d.append(class_names.index(cat))
        else:
            gt_labels_3d.append(-1)
    gt_labels_3d = np.array(gt_labels_3d)
    visualize_camera(
        os.path.join("viz/camera_front", f"{sample_token}.png"),
        image,
        bboxes=bboxes,
        labels=gt_labels_3d,
        transform=info['cams']['CAM_FRONT']['lidar2image'],
        classes=class_names,
    )
    
    # 激光点云bev图绘制3dbox（lidar frame)、车道线
    lidar = np.fromfile(info['lidar_path'], dtype=np.float32)
    lidar = lidar.reshape(-1, 5)
    visualize_lidar(
        os.path.join("viz/lidar", f"{sample_token}.png"),
        lidar,
        bboxes=bboxes,
        points=info['lane_3d'],
        labels=gt_labels_3d,
        xlim=[point_cloud_range[d] for d in [0, 3]],
        ylim=[point_cloud_range[d] for d in [1, 4]],
        classes=class_names,
    )

    # import ipdb; ipdb.set_trace()


def filter_lane_by_dis(global_cor, lane_lidar_dis):
    on_fov = (
        (global_cor[0] < lane_lidar_dis[1])
        & (global_cor[0] >= lane_lidar_dis[0])
        & (global_cor[1] < lane_lidar_dis[3])
        & (global_cor[1] >= lane_lidar_dis[2])
    )#shape:n
    # if on_fov.any():
    #     import ipdb; ipdb.set_trace()
    global_cor = global_cor[:, on_fov]
    
    return global_cor

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

    # camera to lidar
    camera2lidar_rt = np.eye(4).astype(np.float32)
    camera2lidar_rt[:3, :3] = info["sensor2lidar_rotation"].T
    camera2lidar_rt[3, :3] = info["sensor2lidar_translation"]
    info["camera2lidar"] = camera2lidar_rt.T

    # global to lidar transform
    global2lidar = camera2lidar_rt.T @ ego2camera_rt.T @ global2ego_rt.T
    info["global2lidar"] = global2lidar
    global2lidar_tensor = torch.tensor(global2lidar)

    # lidar to camera
    lidar2camera_r = np.linalg.inv(camera2lidar_rt.T[:3, :3])
    lidar2camera_t = (
        camera2lidar_rt[3, :3] @ lidar2camera_r.T
    )
    lidar2camera_rt = np.eye(4).astype(np.float32)
    lidar2camera_rt[:3, :3] = lidar2camera_r.T
    lidar2camera_rt[3, :3] = -lidar2camera_t
    info["lidar2camera"] = lidar2camera_rt.T

    # lidar 2 image
    lidar2image = camera_intrinsics @ lidar2camera_rt.T
    info["lidar2image"] = lidar2image
    lidar2image_tensor = torch.tensor(lidar2image)

    # 将车道线转换到lidar坐标系
    list_lidar_frame_lane = []
    for polygon in map_lane:
        global_cor = torch.tensor(polygon["points"])
        zero_tensor = torch.zeros(global_cor.shape[0], 1)
        global_cor_3d = torch.cat((global_cor, zero_tensor), dim=1).transpose(1, 0)#(3,n)
        lidar_frame_coords = global2lidar_tensor[:3, :3].matmul(global_cor_3d)
        lidar_frame_coords += global2lidar_tensor[:3, 3].reshape(3, 1)# torch.Size([3, n]),将点云转换到lidar坐标系
        list_lidar_frame_lane.append(lidar_frame_coords)

    lane_3d = []
    lane_2d = []
    for global_cor in list_lidar_frame_lane:
        # 筛选lidar周围内的map_lane,注意lidar_frame是右前天
        lane_lidar_dis = [-20.0, 20.0, 0, 60.0]
        global_cor = filter_lane_by_dis(global_cor, lane_lidar_dis)#global_cor是lidar坐标系，和gt_bboxes_3d是一个坐标系下
        if global_cor.numel() == 0:
            continue

        # add z添加Z坐标
        # add_z()

        # project
        points_2d = []
        width, height = img.size

        cur_coords = lidar2image_tensor[:3, :3].matmul(global_cor)
        cur_coords += lidar2image_tensor[:3, 3].reshape(3, 1)# torch.Size([3, n]),将点云转换到图片坐标系

        # get 2d coords
        dist = cur_coords[2, :]
        cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
        cur_coords[:2, :] /= cur_coords[2:3, :]

        draw_points_2d = cur_coords[:2, :].transpose(1, 0)#(n,2)
        depth = cur_coords[2:3, :].transpose(1, 0)

        on_img = (
            (draw_points_2d[..., 0] < width)
            & (draw_points_2d[..., 0] >= 0)
            & (draw_points_2d[..., 1] < height)
            & (draw_points_2d[..., 1] >= 0)
        )

        depth_mask = (depth > 0).squeeze()
        draw_points_2d = draw_points_2d[on_img & depth_mask]
        global_cor = global_cor.transpose(1, 0)[on_img & depth_mask]

        lane_3d.append(global_cor)
        lane_2d.append(draw_points_2d)

    return (lane_3d, lane_2d)

def filter_pkl(info):
    # 1.删除无效字段
    del info['sweeps']
    del info['cams']['CAM_FRONT_RIGHT']
    del info['cams']['CAM_FRONT_LEFT']
    del info['cams']['CAM_BACK']
    del info['cams']['CAM_BACK_LEFT']
    del info['cams']['CAM_BACK_RIGHT']
    # del info['gt_velocity']
    del info['num_radar_pts']

    # 2.根据当前sample token判断属于哪个地图
    sample_token = info['token']
    scene_token = next((entry['scene_token'] for entry in sample_json if entry['token'] == sample_token))  
    log_token = next((entry['log_token'] for entry in scene_json if entry['token'] == scene_token))
    map_token = next((entry['token'] for entry in map_json if log_token in entry['log_tokens']))
    map_name = map_tokens[map_token]
    map_lane = map_locations[map_name]

    # 3.将车道线点投影到前向图片上
    img = Image.open(info['cams']['CAM_FRONT']['data_path'])#rgb
    # img.save('viz/lane/'+sample_token+'.jpg')
    lane_3d, lane_2d = project_lane_to_img(img, map_lane, info['cams']['CAM_FRONT'], sample_token)#返回list[tensor]

    # 4.将能投影到当前图片上的车道线点保存在lane字段中
    info['lane_3d'] = lane_3d #lidar坐标系下
    info['lane_2d'] = lane_2d #前向图片坐标系下

    # draw_2d_points_to_image(img, info)
    # import ipdb; ipdb.set_trace()

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
    info_path = os.path.join(dataroot, 'mmdet3d_nuscenes_front_lane_infos_{}_vel.pkl'.format(info_prefix))

    data_infos = key_infos["infos"]
    filtered_data_infos = []

    for index in tqdm(range(0, len(data_infos))):
    # for index in range(0, len(data_infos)):
        info = data_infos[index]
        # 删除无效字段 & 投影车道线点并添加lane字段
        if filter_pkl(info):
            filtered_data_infos.append(info)        

    # step3: save pkl
    data = dict(infos=filtered_data_infos, metadata=key_infos["metadata"])
    mmcv.dump(data, info_path)