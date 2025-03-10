# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet3d_plugin.models.utils.pe import PE
from mmcv.cnn import ConvModule


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
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ori_gt_bboxes_3d,
                      ori_gt_labels_3d,
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        num_views = len(img_metas)

        proposal_boxes = []
        proposal_scores = []
        proposal_classes = []
        for i in range(num_views):
            proposal_boxes.append(proposal_list[i][:, :6])
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # 根据proposal_boxes位置和车道线估计3dbox最近表面位置、宽度、高度

        # 设置固定长度

        # 设计网络获取观察角度theta_l, 并得到最终的theta(偏航角) = theta_l(观察角度) + theta_ray(arctan(z/x)
        rois = bbox2roi(proposal_list)#给每个框的编码前加一个所属的图片索引,变成5维的向量，最后把所有的cat一下，变成(n,5)的tensor
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 计算损失
        losses = dict()
        results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas)

        cls_scores = results_from_last['pred']['cls_scores']
        bbox_preds = results_from_last['pred']['bbox_preds']

        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        for layer in range(num_layers):
            lw = self.stage_loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses
