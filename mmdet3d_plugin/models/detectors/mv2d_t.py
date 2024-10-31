# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
import os
import cv2

from mmdet3d.core import (bbox3d2result, box3d_multiclass_nms)
from mmdet3d.models.builder import DETECTORS, build_detector, build_head, build_neck
from .mv2d import MV2D


@DETECTORS.register_module()
class MV2DT(MV2D):
    def __init__(self,
                 num_views=6,
                 grad_all=True,
                 **kwargs,
                 ):
        super(MV2DT, self).__init__(**kwargs)
        self.num_views = num_views
        self.grad_all = grad_all#true
    """
    img: #[1, 12, 3, 512, 1408],1-batch_size,12-有12张图片,512-H,1408-W,为什么12张全是sample?其中前6张时当前帧的,后6张是previous
    img_metas: list,img_metas[0].keys()=['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename', 'intrinsics', 'extrinsics', 'timestamp']
                'filename': 图片地址,len=12
                'ori_shape': (900, 1600, 3, 6)
                'img_shape': [(512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3)]
                'lidar2img': len()=12
                'pad_shape': [(512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3), (512, 1408, 3)]
                'scale_factor': 1.0
                'box_mode_3d': <Box3DMode.LIDAR: 0>
                'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
                'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}
                'sample_idx': '80d56e801c7e465995bdb116b3e678aa'
                'pts_filename': './data/nuscenes/samples/LIDAR_TOP/n015-2018-08-03-15-00-36+0800__LIDAR_TOP__1533279682150532.pcd.bin'
                'intrinsics': len()=12
                'extrinsics': len()=12
                timestamp': [0.03807210922241211, 0.030193090438842773, 0.04568791389465332, 0.013006925582885742, 0.003108978271484375, 0.022639036178588867, 0.5380721092224121, 0.5301930904388428, 0.5456879138946533, 0.5130069255828857, 0.5031089782714844, 0.5226390361785889]
    gt_bboxes_2d: list,shape=[1,6],保存的是对应图片上gt标注的左上角和右下角,6应该表示的是6个相机
    gt_labels_2d: 和gt_bboxes_2d相对应的gt的类别,用数字表示
    gt_bboxes_2d_to_3d: 2dbox和3dbox的对应关系,即是gt_bboxes_3d中的哪个目标,保存序号 eg:[0, 2, 5, 6, 7],[],[ 1, -1,  3,  4, -1, -1],[],[ 1, -1,  3, -1, -1, -1, -1],[]
    gt_bboxes_3d: LiDARInstance3DBoxe类型的3dbox
    gt_labels_3d: 和gt_bboxes_3d对应的类别,eg:[tensor([0, 8, 0, 0, 0, 0, 7, 0], device='cuda:0')]
    gt_bboxes_ignore: 
    """
    def forward_train(self,
                      img,   
                      img_metas,
                      gt_bboxes_2d,
                      gt_labels_2d,
                      gt_bboxes_2d_to_3d,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        losses = dict()

        batch_size, num_views, c, h, w = img.shape#[1, 12, 3, 512, 1408]
        img = img.view(batch_size * num_views, *img.shape[2:])#[12, 3, 512, 1408]
        assert batch_size == 1, 'only support batch_size 1 now'

        if self.use_grid_mask:#true
            img = self.grid_mask(img)#一种数据增强方式，生成结构化的格子，然后将格子图像信息删除(填充0)

        # get pseudo monocular input
        # gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas:
        #   independent GT for each view
        # ori_gt_bboxes_3d, ori_gt_labels_3d:
        #   original GT for all the views
        ori_img_metas, ori_gt_bboxes_3d, ori_gt_labels_3d, ori_gt_bboxes_ignore = img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore
        gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas = [], [], [], [], [], []
        for i in range(batch_size):#batch_size==1固定的
            img_metas_views = ori_img_metas[i]#dict
            for j in range(num_views):#假设当前帧12个
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]#(900, 1600, 3)
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)#把ori_img_metas按照图片拆分成list，每个相机对应一个dict

            gt_labels_3d_views = ori_gt_labels_3d[i]#当前帧中所有gt的label
            gt_bboxes_3d_views = ori_gt_bboxes_3d[i].to(gt_labels_3d_views.device)#当前帧中所有gt的box
            for j in range(self.num_views):#6
                gt_ids = (gt_bboxes_2d_to_3d[i][j]).unique()#返回输入张量中所有唯一元素的排序列表,eg:[ 1, -1,  3,  4, -1, -1]->[-1,1,3,4]
                select = gt_ids[gt_ids > -1].long()#过滤掉-1无效值,eg:[1,3,4]
                gt_bboxes_3d.append(gt_bboxes_3d_views[select])#从当前帧中的所有gt中将当前图片对应的gt筛选出来
                gt_labels_3d.append(gt_labels_3d_views[select])
            # no GT in previous frames
            for j in range(self.num_views, num_views):#(6,12)，后6张图片是previous，即前一个关键帧
                box_type = gt_bboxes_3d[0].__class__#mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes
                box_dim = gt_bboxes_3d[0].tensor.size(-1)#9
                gt_bboxes_3d.append(box_type(img.new_zeros((0, box_dim)), box_dim=box_dim))#添加假真值box
                gt_labels_3d.append(img.new_zeros(0, dtype=torch.long))#添加假label

            gt_bboxes.extend(gt_bboxes_2d[i])
            gt_labels.extend(gt_labels_2d[i])
            gt_bboxes_ignore.extend(ori_gt_bboxes_ignore[i])

        # calculate losses for base detector
        if not self.grad_all:
            detector_feat_current = self.extract_feat(img[:self.num_views])
            with torch.no_grad():
                detector_feat_history = self.extract_feat(img[self.num_views:])#历史帧提取特征但不进行梯度更新
            detector_feat = [torch.cat([x1, x2]) for x1, x2 in zip(detector_feat_current, detector_feat_history)]
        else:#
            detector_feat = self.extract_feat(img)#当前帧和前一帧的图片都提取2d特征,self.base_detector.extract_feat(img)
            detector_feat_current = [x[:self.num_views] for x in detector_feat]#档期帧的特征
            detector_feat_history = [x[self.num_views:] for x in detector_feat]#previous帧的特征

        # only the current frame is used in 2D detection loss
        img_current = img[:self.num_views]#当前帧6张图片
        img_metas_current = img_metas[:self.num_views]
        losses_detector = self.base_detector.forward_train_w_feat(
            detector_feat_current,
            img_current,
            img_metas_current,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore)
        for k, v in losses_detector.items():
            losses['det_' + k] = v
        """
        losses:
            'loss_rpn_cls': len()=5,
            'loss_rpn_bbox': len()=5,
            'loss_cls': len()=1,
            'acc': len()=1,
            'loss_bbox': len()=1,
        """
        # generate 2D detection
        with torch.no_grad():#不计算梯度
            self.base_detector.set_detection_cfg(self.train_cfg.get('detection_proposal'))
            results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)#得到每个类别的检测结果，[num_boxes, 5->(x1, y1, x2, y2, score)]
            detections = self.process_2d_detections(results, img.device)#检测得到的2dbox根据参数中的min_bbox_size过滤

        if self.train_cfg.get('complement_2d_gt', -1) > 0:#0.4,使用gt进行补偿？
            detections_gt = self.process_2d_gt(gt_bboxes, gt_labels, img.device)#[num_boxes, 4->(x1, y1, x2, y2)]->[num_boxes, 6->(x1, y1, x2, y2, 1, label)]
            detections_gt = detections_gt + [img.new_zeros((0, 6))] * (num_views - self.num_views)#后面6张图片添加假gt
            detections = [self.complement_2d_gt(det, det_gt, thr=self.train_cfg.get('complement_2d_gt'))
                          for det, det_gt in zip(detections, detections_gt)]#依次对每张图片，把漏检的gt和det cat到一起

        # calculate losses for 3d detector
        if not self.grad_all:
            feat_current = self.process_detector_feat(detector_feat_current)
            with torch.no_grad():
                feat_history = self.process_detector_feat(detector_feat_history)
            feat = [torch.cat([x1, x2]) for x1, x2 in zip(feat_current, feat_history)]
        else:#
            feat = self.process_detector_feat(detector_feat)#neck网络

        roi_losses = self.roi_head.forward_train(feat, img_metas, detections,                           # num_views
                                                 gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,      # self.num_views
                                                 ori_gt_bboxes_3d, ori_gt_labels_3d,
                                                 attr_labels, None)#MV2DTHead
        losses.update(roi_losses)
        """
        roi_losses:
            'l0.dn_loss_cls', 'l0.dn_loss_bbox', 'l1.dn_loss_cls', 'l1.dn_loss_bbox', 
            'l2.dn_loss_cls', 'l2.dn_loss_bbox', 'l3.dn_loss_cls', 'l3.dn_loss_bbox', 
            'l4.dn_loss_cls', 'l4.dn_loss_bbox', 'l5.dn_loss_cls', 'l5.dn_loss_bbox', 
            'l0.loss_cls', 'l0.loss_bbox', 'l1.loss_cls', 'l1.loss_bbox', 
            'l2.loss_cls', 'l2.loss_bbox', 'l3.loss_cls', 'l3.loss_bbox', 
            'l4.loss_cls', 'l4.loss_bbox', 'l5.loss_cls', 'l5.loss_bbox
        """
        return losses

