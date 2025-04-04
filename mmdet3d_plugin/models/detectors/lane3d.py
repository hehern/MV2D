# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
import os
import cv2

from mmdet.models.builder import DETECTORS, build_detector, build_head, build_neck
from mmdet3d.core import (bbox3d2result, box3d_multiclass_nms)
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d_plugin.models.utils.grid_mask import CustomGridMask


@DETECTORS.register_module()
class LANE3D(Base3DDetector):

    def __init__(self,
                 base_detector,
                 neck,
                 roi_head,
                 train_cfg=None,
                 test_cfg=None,
                 use_grid_mask=None,
                 init_cfg=None,
                 num_views=1,
                 **kwargs,
                 ):
        super(Base3DDetector, self).__init__(init_cfg)

        self.base_detector = build_detector(base_detector)#TwoStageDetBase
        self.neck = build_neck(neck)#FPN
        if train_cfg is not None:
            roi_head.update(train_cfg=train_cfg['rcnn'])
        if test_cfg is not None:
            roi_head.update(test_cfg=test_cfg['rcnn'])
        self.roi_head = build_head(roi_head)

        self.use_grid_mask = isinstance(use_grid_mask, dict)#true
        if self.use_grid_mask:
            self.grid_mask = CustomGridMask(**use_grid_mask)#这是啥？

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_views = num_views

    def process_2d_gt(self, gt_bboxes, gt_labels, device):
        """
        :param gt_bboxes:
            gt_bboxes: list[boxes] of size BATCH_SIZE
            boxes: [num_boxes, 4->(x1, y1, x2, y2)]
        :param gt_labels:
        :return:
        """
        return [torch.cat(
            [bboxes.to(device), torch.ones([len(labels), 1], dtype=bboxes.dtype, device=device),
             labels.unsqueeze(-1).to(bboxes.dtype)], dim=-1).to(device)
                for bboxes, labels in zip(gt_bboxes, gt_labels)]

    def process_2d_detections(self, results, device):
        """
        :param results:
            results: list[per_cls_res] of size BATCH_SIZE
            per_cls_res: list(boxes) of size NUM_CLASSES
            boxes: ndarray of shape [num_boxes, 5->(x1, y1, x2, y2, score)]
        :return:
            detections: list[ndarray of shape [num_boxes, 6->(x1, y1, x2, y2, score, label_id)]] of size len(results)
        """
        detections = [torch.cat(
            [torch.cat([torch.tensor(boxes), torch.full((len(boxes), 1), label_id, dtype=torch.float)], dim=1) for
             label_id, boxes in
             enumerate(res)], dim=0).to(device) for res in results]

        if self.train_cfg is not None:
            min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        else:
            min_bbox_size = self.test_cfg['detection_proposal'].get('min_bbox_size', 0)
        if min_bbox_size > 0:
            new_detections = []
            for det in detections:
                wh = det[:, 2:4] - det[:, 0:2]
                valid = (wh >= min_bbox_size).all(dim=1)
                new_detections.append(det[valid])
            detections = new_detections

        return detections

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]                # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]             # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))     # [*, n, m, 2]
        intersect = wh.prod(-1)                                         # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]      # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]      # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

    def complement_2d_gt(self, detections, gts, thr=0.35):
        # detections: [n, 6], gts: [m, 6]
        if len(gts) == 0:
            return detections
        if len(detections) == 0:
            return gts
        iou = self.box_iou(gts, detections)
        max_iou = iou.max(-1)[0]
        complement_ids = max_iou < thr
        min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        wh = gts[:, 2:4] - gts[:, 0:2]
        valid_ids = (wh >= min_bbox_size).all(dim=1)
        complement_gts = gts[complement_ids & valid_ids]
        return torch.cat([detections, complement_gts], dim=0)

    def extract_feat(self, img):
        return self.base_detector.extract_feat(img)

    def process_detector_feat(self, detector_feat):
        if self.with_neck:
            feat = self.neck(detector_feat)
        else:
            feat = detector_feat
        return feat

    def forward_train(self,
                      img,          #图片，tensor, [batch_size, 1, 3, 512, 1408]
                      img_metas,    #原始信息，list,len=4,keys:'num_views', 'filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'pts_filename', 'intrinsics', 'extrinsics', 'timestamp'
                      gt_bboxes_2d, #2d标注(每个batch内依次所有相机上的gt），list[list]，tensor，len()=batch_size, len(0)=num_views
                      gt_labels_2d, #2d标注类别，list[list],tensor
                      gt_bboxes_2d_to_3d, #2d到3d目标映射表，list[list],tensor
                      gt_bboxes_3d,       #3d标注框(当前帧所有的），list[LiDARInstance3DBoxes]，len()=bs
                      gt_labels_3d,       #3d标注类别,list[tensor],len()=bs
                      attr_labels=None,
                      gt_bboxes_ignore=None):

        losses = dict()

        batch_size, num_views, c, h, w = img.shape#4,1,3,512,1408
        img = img.view(batch_size * num_views, *img.shape[2:])#[4, 3, 512, 1408]
        assert num_views == 1, 'only support front camera now'

        if self.use_grid_mask:#true
            img = self.grid_mask(img)

        # step1: get pseudo monocular input
        # gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas:
        #   independent GT for front camera
        ori_img_metas, ori_gt_bboxes_3d, ori_gt_labels_3d, ori_gt_bboxes_ignore = img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore
        gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore, img_metas = [], [], [], [], [], []
        for i in range(batch_size):
            img_metas_views = ori_img_metas[i]
            for j in range(num_views):
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if k == 'lane_2d' or k == 'lane_3d':
                        img_meta[k] = v
                    elif isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)

            gt_labels_3d_views = ori_gt_labels_3d[i]
            gt_bboxes_3d_views = ori_gt_bboxes_3d[i].to(gt_labels_3d_views.device)
            for j in range(self.num_views):
                gt_ids = (gt_bboxes_2d_to_3d[i][j]).unique()
                select = gt_ids[gt_ids > -1].long()
                gt_bboxes_3d.append(gt_bboxes_3d_views[select])
                gt_labels_3d.append(gt_labels_3d_views[select])

            gt_bboxes.extend(gt_bboxes_2d[i])
            gt_labels.extend(gt_labels_2d[i])
            gt_bboxes_ignore.extend(ori_gt_bboxes_ignore[i])

        # step2: extract 2d feature
        detector_feat = self.extract_feat(img)#提取2d特征

        # step3: calculate 2D detection loss
        losses_detector = self.base_detector.forward_train_w_feat(
            detector_feat,
            img,
            img_metas,
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
        
        # step4: generate 2D detection
        with torch.no_grad():#不计算梯度
            self.base_detector.set_detection_cfg(self.train_cfg.get('detection_proposal'))
            results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)#得到每个类别的检测结果，[num_boxes, 5->(x1, y1, x2, y2, score)]
            detections = self.process_2d_detections(results, img.device)##检测得到的2dbox根据参数中的min_bbox_size过滤

        if self.train_cfg.get('complement_2d_gt', -1) > 0:#0.4,使用gt进行补偿
            detections_gt = self.process_2d_gt(gt_bboxes, gt_labels, img.device)#[num_boxes, 4->(x1, y1, x2, y2)]->[num_boxes, 6->(x1, y1, x2, y2, 1, label)]
            detections = [self.complement_2d_gt(det, det_gt, thr=self.train_cfg.get('complement_2d_gt'))
                          for det, det_gt in zip(detections, detections_gt)]#依次对每张图片，把漏检的gt和det cat到一起

        # step5: extract feature
        feat = self.process_detector_feat(detector_feat)#neck网络,feat tuple,feat[0].shape=[bs, 256, 32, 88]

        # step6: calculate losses for lane_3d detector
        roi_losses = self.roi_head.forward_train(feat, img, img_metas, detections,                           # num_views
                                                 gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d,      # self.num_views
                                                 attr_labels, None)#LANE3DHead

        losses.update(roi_losses)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        num_augs = len(img)#1
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_metas)))

        if num_augs == 1:#true
            return self.simple_test(img[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, proposal_bboxes=None, proposal_labels=None, rescale=False, **kwargs):

        # process multi-view inputs
        batch_size, num_views, c, h, w = img.shape#[1,1,3,512,1408]
        img = img.view(batch_size * num_views, c, h, w)#[1,3,512,1408]
        ori_img_metas = img_metas#list[dict]
        img_metas = []
        gt_bboxes, gt_labels = [], []
        for i in range(batch_size):#1
            img_metas_views = ori_img_metas[i]
            for j in range(num_views):
                img_meta = dict(num_views=num_views)
                for k, v in img_metas_views.items():
                    if k == 'lane_2d' or k == 'lane_3d':
                        img_meta[k] = v
                    elif isinstance(v, list):
                        img_meta[k] = v[j]
                    elif k == 'ori_shape':
                        img_meta[k] = v[:3]
                    else:
                        img_meta[k] = v
                img_metas.append(img_meta)
            if proposal_bboxes is not None:#为什么参数没有携带proposal_bboxes？？？？
                gt_bboxes.extend(proposal_bboxes[i])
                gt_labels.extend(proposal_labels[i])

        detector_feat = self.extract_feat(img)

        # generate 3D detection
        self.base_detector.set_detection_cfg(self.test_cfg.get('detection_proposal'))
        det_results = self.base_detector.simple_test_w_feat(detector_feat, img_metas)
        detections = self.process_2d_detections(det_results, device=img.device)

        feat = self.process_detector_feat(detector_feat)

        # generate 3D detection
        bbox_outputs_all = self.roi_head.simple_test(feat, detections, img_metas, rescale=rescale)
        bbox_outputs = []
        box_type_3d = img_metas[0]['box_type_3d']

        # 3D NMS
        for i in range(batch_size):#1
            # bbox_outputs_i: len(num_views)
            bbox_outputs_i = bbox_outputs_all[i * num_views:i * num_views + num_views]
            all_bboxes = box_type_3d.cat([x[0] for x in bbox_outputs_i])
            all_scores = torch.cat([x[1] for x in bbox_outputs_i])
            all_classes = torch.cat([x[2] for x in bbox_outputs_i])

            all_scores_classes = all_scores.new_zeros(
                (len(all_scores), self.roi_head.num_classes + 1)).scatter_(1, all_classes[:, None], all_scores[:, None])

            cfg = self.test_cfg.get('rcnn')
            results = box3d_multiclass_nms(all_bboxes.tensor, all_bboxes.bev,
                                           all_scores_classes, cfg.score_thr, cfg.max_per_scene, cfg.nms)

            bbox_outputs.append((
                box_type_3d(results[0], box_dim=all_bboxes.tensor.shape[1], with_yaw=all_bboxes.with_yaw),
                results[1], results[2]))

        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_outputs
        ]

        bbox_list = [dict() for i in range(batch_size)]
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError