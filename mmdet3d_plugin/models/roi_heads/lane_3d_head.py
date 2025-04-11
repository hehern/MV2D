# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS, build_head, build_roi_extractor

import numpy as np

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
def draw_proposal_on_img(proposal_boxes, img, img_metas, closest_result_pair_list_bs = None):
    from tools.visualize import visualize_camera
    import os
    img_cpu = img.cpu().numpy()#(bs, 3, 512, 1408) bs*C*H*W
    img_hwc = img_cpu.transpose(0,2,3,1)#(bs, 512, 1408, 3)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img_hwc *= std
    img_hwc += mean#反ImageNormalize操作,rgb顺序
    img_hwc = img_hwc[..., [2, 1, 0]]#转换为bgr顺序

    for i in range(img.shape[0]):#bs
        # 绘制车道线
        from PIL import Image, ImageDraw
        img_single = Image.fromarray(np.uint8(img_hwc[i]))
        draw = ImageDraw.Draw(img_single)
        point_color = (255, 0, 0)
        for points_2d in img_metas[i]['lane_2d']:
            for (x, y) in points_2d:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=point_color)

        # 绘制距离box最近的车道线点之间的连线
        if closest_result_pair_list_bs is not None:
            if len(closest_result_pair_list_bs[i]) != 0:
                for tmp in closest_result_pair_list_bs[i]:
                    tensor_idx, j, k, box_id = tmp
                    # import ipdb; ipdb.set_trace()
                    x, y = img_metas[i]['lane_2d'][tensor_idx][j, :]
                    box = proposal_boxes[i][box_id]
                    draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 255, 0))
                    draw.ellipse((box[2].cpu()-5, box[3].cpu()-5, box[2].cpu()+5, box[3].cpu()+5), fill=(0, 255, 0))
                    draw.line([(x, y), (box[2].cpu(), box[3].cpu())], fill=(0, 255, 0), width=2)
                    # next_x, next_y = img_metas[i]['lane_2d'][tensor_idx][k, :]
                    # draw.line([(x, y), (next_x, next_y)], fill=(0, 255, 255), width=2)

        img_single = np.array(img_single).astype(np.float32)

        # 绘制pred 2d box
        sample_token = img_metas[i]['sample_idx']
        visualize_camera(
            os.path.join("viz/head_draw_proposal_on_img", f"{sample_token}.png"),
            img_single,
            bboxes_2d=proposal_boxes[i].cpu().numpy(),
            classes=class_names,
        )
    # import ipdb; ipdb.set_trace()

@HEADS.register_module()
class LANE3DHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    def __init__(self,
                 bbox_roi_extractor,
                 bbox_head,
                 pc_range,
                 force_fp32=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(LANE3DHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head,
                                       train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        self.roi_size = bbox_roi_extractor['roi_layer']['output_size']#7
        if isinstance(self.roi_size, int):
            self.roi_size = [self.roi_size, self.roi_size]#[7, 7]

        self.pc_range = pc_range#[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.stage_loss_weights = train_cfg.get('stage_loss_weights') if train_cfg else None
        self.force_fp32 = force_fp32

    @torch.no_grad()
    def get_box_params(self, bboxes, intrinsics, extrinsics):
        # TODO: check grad flow from boxes to intrinsic
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = torch.from_numpy(intrinsic).to(bbox.device).double()
            extrinsic = torch.from_numpy(extrinsic).to(bbox.device).double()
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            # consider corners
            wh_bbox = bbox[:, 2:4] - bbox[:, :2]
            wh_roi = wh_bbox.new_tensor(self.roi_size)
            scale = wh_roi[None] / wh_bbox
            intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
            intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list

    @property
    def num_classes(self):
        return self.bbox_head.num_classes

    def init_assigner_sampler(self):
        self.bbox_assigner = None
        self.bbox_sampler = None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        bbox_head.update(dict(train_cfg=self.train_cfg, test_cfg=self.test_cfg))
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        raise NotImplementedError

    def forward_train(self,
                      x,            #图像特征,tuple(torch.Tensor),feat[0].shape=[bs, 256, 32, 88]
                      img,          #原始图片,[bs, 3, 512, 1408]
                      img_metas,    #原始数据信息len(img_metas)=bs,['num_views', 'filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename', 'intrinsics', 'extrinsics', 'timestamp']
                      proposal_list,#2d pred,list,len()=bs
                      gt_bboxes,    #2d gt-按照图片存放
                      gt_labels,    #2d gt label
                      gt_bboxes_3d, #3d gt
                      gt_labels_3d, #3d gt label
                    #   lane_2d,      #2d车道线,list,len=bs,
                    #   lane_3d,      #2d车道线，lidar frame下，右前天坐标系
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        bs = x[0].shape[0]#bs
        num_views = img_metas[0]['num_views']#1
        assert num_views == 1, 'only support front camera now'

        proposal_boxes = []#bs数据中每张图片上的2dbox
        proposal_scores = []#score
        proposal_classes = []#label(数字)
        for i in range(bs):
            proposal_boxes.append(proposal_list[i][:, :6])#[x1, y1, x2, y2, score, label]
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

                # avoid empty 2D detection
        if sum([len(p) for p in proposal_boxes]) == 0:#2D检测框为0的情况下
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_boxes[0].dtype,
                                    device=proposal_boxes[0].device)
            proposal_boxes = [proposal] + proposal_boxes[1:]#填充假的

        # step1: 根据proposal_boxes位置和车道线估计3dbox最近表面位置、宽度、高度
        # 1.1 坐标进行插值
        pos_xy_wh, closest_result_pair_list_bs = self._interpolation_get_pos(proposal_boxes, img_metas)
        # 1.2 将proposal_boxes以及插值结果可视化在图片上
        # draw_proposal_on_img(proposal_boxes, img, img_metas, closest_result_pair_list_bs)

        # step2: 网络获取theta_l, 并得到最终的theta(偏航角) = theta_l + theta_ray(arctan(z/x)
        losses = dict()

        # step3: 输入类别、xy坐标、宽度、高度，估计位置误差、长度、宽度误差、高度误差
        bbox_results = self._bbox_forward(x, proposal_boxes, pos_xy_wh)

        # step4: use the matching results for loss calculation
        loss_bbox = self.bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, bbox_results,
        )
        for k, v in loss_bbox.items():
            losses['3d_' + k] = v

        return losses

    def _interpolation_get_pos(self, proposal_boxes, img_metas):
        # bs = len(img_metas)
        # lane_3d = []
        # for index in range(bs):
        #     concatenated_tensor = torch.cat(img_metas[index]['lane_3d'], dim=0)
        #     lane_3d.append(concatenated_tensor)
        # lane_3d = torch.cat(lane_3d, dim=0)
        # if torch.isnan(lane_3d).any():
        #     print(lane_3d)
        # assert not torch.isnan(lane_3d).any(), 'img_metas lane_3d has nan data'

        """对proposal_boxes右下角坐标进行插值处理得到位置xy"""
        proposal_boxes_3d_bs = []
        closest_result_pair_list_bs = []
        img_h = img_metas[0]['img_shape'][0]
        for i in range(len(proposal_boxes)):#bs
            bboxes_2d = proposal_boxes[i].cpu().numpy()
            proposal_boxes_3d = []
            closest_result_pair_list = []
            if bboxes_2d is not None and len(bboxes_2d) > 0:
                for box_id, box in enumerate(bboxes_2d):
                    right_down = np.array([box[2], box[3]])
                    result_pairs = []
                    #筛选出将box右下角坐标夹在图像纵向（y坐标）的lane_2d点对，再从中筛选出横向（x坐标）距离最近的
                    for tensor_idx, tensor in enumerate(img_metas[i]['lane_2d']):
                        points = tensor.numpy()
                        n = points.shape[0]
                        for j in range(n):
                            k = (j+1)%n
                            _, y1 = points[j]
                            _, y2 = points[k]
                            # 检查是否满足 y 坐标将 right_down 的 y 坐标夹在中间的条件
                            if y1 <= right_down[1] < y2 or y2 <= right_down[1] < y1:
                                # 将点对添加到结果列表中
                                result_pairs.append((tensor_idx, j, k))
                    if not result_pairs:
                        #没有找到点对:如果是比较近的点就赋值很近，如果是远的点就赋值max,其余情况可能是没有车道线就暂时舍弃掉
                        if right_down[1] > (img_h-10):#很近的地方
                            proposal_boxes_3d.append(torch.tensor([box_id, 0.0, self.pc_range[1], 1.0, 1.0]))
                        elif right_down[1] < (10):#很远的地方
                            proposal_boxes_3d.append(torch.tensor([box_id, 0.0, self.pc_range[4], 1.0, 1.0]))
                        else:#填充假的
                            proposal_boxes_3d.append(torch.tensor([box_id, 0.0, self.pc_range[4], 1.0, 1.0]))

                    else:
                        # 初始化最小距离和对应的序号
                        min_distance = float('inf')
                        closest_idx = -1
                        closest_result_pair = None
                        for tensor_idx, j, k in result_pairs:
                            # 提取对应的 tensor 中的点
                            point = img_metas[i]['lane_2d'][tensor_idx][j, :] #这里的 j 是点的索引，k 可能在这个场景下不用
                            # 计算 x 轴上的距离
                            distance_x = abs(float(point[0]) - float(right_down[0]))
                            # 更新最小距离和对应的序号
                            if distance_x < min_distance:
                                min_distance = distance_x
                                closest_idx = tensor_idx  # 这里保存的是 tensor 的索引，如果需要保存整体的序号 (tensor_idx, j, k)，则保存 result_pair
                                closest_result_pair = (tensor_idx, j, k)
                        if closest_idx != -1:
                            closest_result_pair_list.append(closest_result_pair+(box_id,))

                            tensor_idx, j, k = closest_result_pair
                            y1 = img_metas[i]['lane_2d'][tensor_idx][j][1]
                            y2 = img_metas[i]['lane_2d'][tensor_idx][k][1]
                            x1 = img_metas[i]['lane_2d'][tensor_idx][j][0]
                            x2 = img_metas[i]['lane_2d'][tensor_idx][k][0]
                            y1_3d = img_metas[i]['lane_3d'][tensor_idx][j][1]
                            y2_3d = img_metas[i]['lane_3d'][tensor_idx][k][1]
                            x1_3d = img_metas[i]['lane_3d'][tensor_idx][j][0]
                            x2_3d = img_metas[i]['lane_3d'][tensor_idx][k][0]
                            
                            box_middle_x = (box[0] + box[2]) / 2.0
                            y_3d = (right_down[1] - y1) / (y2 - y1) * (y2_3d - y1_3d) + y1_3d
                            x_3d = (box_middle_x - x1) / (x2 - x1) * (x2_3d - x1_3d) + x1_3d
                            width = abs((box[0] - box[2]) / (x2 - x1) * (x2_3d - x1_3d))
                            height = abs((box[1] - box[3]) / (box[0] - box[2])) * width
                            # import ipdb; ipdb.set_trace()
                            if math.isnan(float(x_3d)):
                                print('x_3d nan')
                            if math.isnan(float(y_3d)):
                                print('y_3d nan')
                            if math.isnan(float(width)):
                                print('width nan')
                            if math.isnan(float(height)):
                                print('height nan')
                            # print("x_3d = " + str(x_3d) + ", y_3d = " + str(y_3d) + ", width = " + str(width) + ", height = " + str(height))
                            proposal_boxes_3d.append(torch.tensor([box_id, float(x_3d), float(y_3d), float(width), float(height)]))
                        else:
                            proposal_boxes_3d.append(torch.tensor([box_id, self.pc_range[3], self.pc_range[4], 1.0, 1.0]))
            proposal_boxes_3d_bs.append(proposal_boxes_3d)
            closest_result_pair_list_bs.append(closest_result_pair_list)
        return proposal_boxes_3d_bs, closest_result_pair_list_bs
    
    def _bbox_forward(self, 
                      x,              #tuple(tensor),len()=1,tensor.shape=[bs, 256, 32, 88]
                      proposal_list,  #list(tensor),len()=bs,tensor.shape=[n,6]
                      pos_xy_wh):

        rois = bbox2roi(proposal_list)#给每个框的编码前加一个所属的图片索引,变成5维的向量，最后把所有的cat一下，变成(sum(n),5)的tensor

        # 把不同size的候选框从原图上映射到选中的feature map上，然后转化为设定好的等长向量(譬如7x7),x0[[12, 256, 32, 88]],eg:bbox_feats[sum(n), 256, 7, 7]
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 估计box误差(sigma_x, sigma_y, sigma_w, l, z, sigma_h, theta_local)
        all_cls_scores, all_bbox_preds = self.bbox_head(bbox_feats)#[sum(n), 7]

        # 封装bbox_preds
        tensors_pos_xy_wh = [tensor for sublist in pos_xy_wh for tensor in sublist]
        # print("all_bbox_preds.shape[0] = " + str(all_bbox_preds.shape[0]) + ", len tensors_pos_xy_wh = " + str(len(tensors_pos_xy_wh)))
        pos_xy_wh = torch.stack(tensors_pos_xy_wh, dim=0).to(all_bbox_preds.device)
        assert pos_xy_wh.shape[0] == all_bbox_preds.shape[0], 'The number of 3d_boxes and 2d_boxes is not equal.'
        assert not torch.isnan(pos_xy_wh).any(), 'pos_xy_wh has nan data'
        new_columns = torch.cat((all_bbox_preds[:, 6:7], torch.zeros(all_bbox_preds.shape[0], 2, dtype=all_bbox_preds.dtype, device=all_bbox_preds.device)), dim=1)
        all_bbox_preds = torch.cat((all_bbox_preds, new_columns), dim=1)
        all_bbox_preds[:, 0] += pos_xy_wh[:, 1]
        all_bbox_preds[:, 1] += pos_xy_wh[:, 2]
        all_bbox_preds[:, 2] += pos_xy_wh[:, 3]
        all_bbox_preds[:, 5] += pos_xy_wh[:, 4]
        all_bbox_preds[:, 6] = (all_bbox_preds[:, 6] + torch.atan2(pos_xy_wh[:, 2], pos_xy_wh[:, 1])).sin()
        all_bbox_preds[:, 7] = (all_bbox_preds[:, 7] + torch.atan2(pos_xy_wh[:, 2], pos_xy_wh[:, 1])).cos()

        group_sizes = [p.shape[0] for p in proposal_list]
        all_cls_scores_bs = torch.split(all_cls_scores, group_sizes, dim=0)
        proposal_boxes_3d_bs = torch.split(all_bbox_preds, group_sizes, dim=0)

        # import ipdb; ipdb.set_trace()
        # 返回结果
        bbox_results = dict(
            cls_scores=all_cls_scores_bs,#a按照bs分成list
            bbox_preds=proposal_boxes_3d_bs,
        )

        return bbox_results
    
    def simple_test(self, x, proposal_list, img, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        if sum([len(p) for p in proposal_list]) == 0:#2D检测框为0的情况下
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]#填充假的

        pos_xy_wh, closest_result_pair_list_bs = self._interpolation_get_pos(proposal_list, img_metas)
        # draw_proposal_on_img(proposal_list, img, img_metas, closest_result_pair_list_bs)
        bbox_results = self._bbox_forward(x, proposal_list, pos_xy_wh)

        cls_scores = bbox_results['cls_scores'][-1]
        bbox_preds = bbox_results['bbox_preds'][-1]

        bbox_list = self.bbox_head.get_bboxes({'cls_scores': [cls_scores], 'bbox_preds': [bbox_preds]}, img_metas,)

        return bbox_list