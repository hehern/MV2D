# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmdet.core import build_bbox_coder, build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d_plugin.models.utils.pe import pos2posemb3d
from mmdet3d_plugin.models.utils import PETRTransformer


@HEADS.register_module()
class LANEBoxHead(BaseModule):
    def __init__(self,
                 num_classes,
                 num_reg_fcs=2,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 embed_dims=256,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 bbox_coder=dict(
                     type='NMSFreeCoder',
                     # type='NMSFreeClsCoder',
                     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                     max_num=100,
                     num_classes=10),
                 sync_cls_avg_factor=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super(LANEBoxHead, self).__init__()

        self.loss_cls = build_loss(loss_cls)#分类损失
        self.loss_bbox = build_loss(loss_bbox)#box损失
        self.pc_range = pc_range
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.embed_dims = embed_dims

        # 输出：sigma_x, sigma_y, sigma_w, l, z, sigma_h, theta_local
        self.reg_branch = nn.Sequential(
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 7)
        )
        self.cls_branch = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),  # 将多维输入一维化
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )#分类网络

        # follow PETR
        self.bbox_coder = build_bbox_coder(bbox_coder)#NMSFreeCoder-非极大值抑制
        if train_cfg is not None:
            self.assigner = build_assigner(train_cfg.get('assigner'))#HungarianAssigner3D-匈牙利匹配
            self.sampler = build_sampler(train_cfg.get('sampler_cfg', dict(type='PseudoSampler')))
        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            if 'class_weight' in loss_cls:
                loss_cls.pop('class_weight')
            self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if 'code_weights' in kwargs:
            self.code_weights = kwargs['code_weights']
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]#这是什么权重？
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

    def init_weights(self):#初始化权重
        """Initialize the transformer weights."""
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def forward(self, x):
        all_cls_scores = self.cls_branch(x)
        x = x.view(-1, 256 * 7 * 7)
        all_bbox_preds = self.reg_branch(x)
        all_bbox_preds[..., 6] = all_bbox_preds[..., 6].sigmoid()
        all_bbox_preds[..., 6] = all_bbox_preds[..., 6] * 2 * math.pi - math.pi#角度转换为-pi-pi
        return all_cls_scores, all_bbox_preds

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)#7
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]#[n,7]
        bbox_weights = torch.zeros_like(bbox_pred)#[n,10]
        bbox_weights[pos_inds] = 1.0
       # DETR
        if sampling_result.pos_gt_bboxes.shape[1] == 4:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes.reshape(sampling_result.pos_gt_bboxes.shape[0],
                                                                           self.code_size - 1)
        else:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'gt_bboxes_list', 'gt_labels_list'))
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    cls_reg_targets=None,
                    gt_bboxes_ignore_list=None):

        num_imgs = len(cls_scores)#bs
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]#按照bs，分成list
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        if cls_reg_targets is None:
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                               gt_bboxes_list, gt_labels_list,
                                               gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        cls_scores, bbox_preds = torch.cat(cls_scores), torch.cat(bbox_preds)
        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        # loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        if len(cls_scores) == 0:
            loss_cls = cls_scores.sum() * cls_avg_factor
        else:
            loss_cls = self.loss_cls(cls_scores, labels, weight=label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, None)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, cls_reg_targets
    
    def loss(self,
             gt_bboxes_3d_list,     #(x, y, z, x_size, y_size, z_size, yaw, vx, vy)
             gt_labels_3d_list,
             preds_dicts,
             cls_reg_targets=None,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        
        cls_scores = preds_dicts['cls_scores']#list[tensor],len=bs,tensor.shape=[n,10],n表示当前帧有n个目标，10为类别score
        bbox_preds = preds_dicts['bbox_preds']#list[tensor],len=bs,tensor.shape=[n,10]


        device = gt_labels_3d_list[0].device#cuda
        gt_bboxes_3d_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_3d_list]#3d_box转换为前面三维使用中心点坐标

        losses_cls, losses_bbox, cls_reg_targets = self.loss_single(
            cls_scores, bbox_preds, gt_bboxes_3d_list, gt_labels_3d_list,
            cls_reg_targets=cls_reg_targets, gt_bboxes_ignore_list=gt_bboxes_ignore)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls
        loss_dict['loss_bbox'] = losses_bbox
        # loss_dict['cls_reg_targets'] = cls_reg_targets
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list