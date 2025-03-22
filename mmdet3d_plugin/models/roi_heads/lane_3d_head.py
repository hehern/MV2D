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

import numpy as np

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
def draw_proposal_on_img(proposal_boxes, img, lane_2d):
    from tools.visualize import visualize_camera
    import os
    img_cpu = img.cpu().numpy()#(bs, 3, 512, 1408) bs*C*H*W
    img_hwc = img_cpu.transpose(0,2,3,1)#(bs, 512, 1408, 3)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img_hwc *= std
    img_hwc += mean#反ImageNormalize操作,rgb顺序
    img_hwc = img_hwc[..., [2, 1, 0]]#转换为bgr顺序

    for i in range(img.shape[0]):
        # 绘制车道线
        from PIL import Image, ImageDraw
        img_single = Image.fromarray(np.uint8(img_hwc[i]))
        draw = ImageDraw.Draw(img_single)
        point_color = (255, 0, 0)
        for points_2d in lane_2d[i]:
            for (x, y) in points_2d:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=point_color)
        img_single = np.array(img_single).astype(np.float32)

        # 绘制pred 2d box
        visualize_camera(
            os.path.join("viz/head_draw_proposal_on_img", f"{i}.png"),
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
                      lane_2d,      #2d车道线,list,len=bs,
                      lane_3d,      #2d车道线，lidar frame下，右前天坐标系
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

        # step1: 根据proposal_boxes位置和车道线估计3dbox最近表面位置、宽度、高度
        
        # 1.1 坐标进行插值，注意这里使用的是
        pos_xy, width = self._interpolation_get_pos(proposal_boxes, lane_2d, lane_3d, img.shape[2], img.shape[3])
        # 1.2 将proposal_boxes以及插值结果可视化在图片上
        draw_proposal_on_img(proposal_boxes, img, lane_2d)

        # step2: 设计网络获取观察角度theta_l, 并得到最终的theta(偏航角) = theta_l(观察角度) + theta_ray(arctan(z/x)
        losses = dict()

        # step3: 输入类别、前向坐标、xy坐标、宽度、高度，估计位置误差、长度、宽度误差、高度误差
        results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas)

        preds = results_from_last['pred']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)#[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # step4: use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        if results_from_last.get('dn_mask_dict', None) is not None:
            dn_mask_dict = results_from_last['dn_mask_dict']
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_dn_loss(
                dn_mask_dict)
            for i in range(len(output_known_class)):
                dn_loss_cls, dn_loss_bbox = self.bbox_head.dn_loss_single(
                    output_known_class[i], output_known_coord[i], known_bboxs, known_labels, num_tgt,
                    self.pc_range, self.denoise_split, neg_bbox_loss=self.neg_bbox_loss
                )
                losses[f'l{i}.dn_loss_cls'] = dn_loss_cls * self.denoise_weight * loss_weights[i]
                losses[f'l{i}.dn_loss_bbox'] = dn_loss_bbox * self.denoise_weight * loss_weights[i]

        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses

    def _interpolation_get_pos(self, proposal_boxes, lane_2d, lane_3d, img_h, img_w):
        """对proposal_boxes右下角坐标进行插值处理得到位置xy"""
        import ipdb; ipdb.set_trace()
        proposal_boxes_3d_bs = []
        for i in range(len(proposal_boxes)):#bs
            bboxes_2d = proposal_boxes[i].cpu().numpy()
            if bboxes_2d is not None and len(bboxes_2d) > 0:
                proposal_boxes_3d = []
                for box_id, box in enumerate(bboxes_2d):
                    right_down = np.array([box[2], box[3]])
                    result_pairs = []
                    #筛选出将box右下角坐标夹在图像纵向（y坐标）的lane_2d点对，再从中筛选出横向（x坐标）距离最近的
                    for tensor_idx, tensor in enumerate(lane_2d[i]):
                        points = tensor.numpy()
                        n = points.shape[0]
                        for j in range(n):
                            k = (i+1)%n
                            _, y1 = points[j]
                            _, y2 = points[k]
                            # 检查是否满足 y 坐标将 right_down 的 y 坐标夹在中间的条件
                            if y1 <= right_down[1] < y2 or y2 <= right_down[1] < y1:
                                # 将点对添加到结果列表中
                                result_pairs.append((tensor_idx, j, k))
                    if not result_pairs:
                        #没有找到点对:如果是比较近的点就赋值很近，如果是远的点就赋值max,其余情况可能是没有车道线就暂时舍弃掉
                        if right_down[1] > (img_h-10):#很近的地方
                            proposal_boxes_3d.append((box_id, x_3d, 0.0))
                        elif right_down[1] < (10):#很远的地方
                            proposal_boxes_3d.append((box_id, x_3d, 50.0))

                    else:
                        # 初始化最小距离和对应的序号
                        min_distance = float('inf')
                        closest_idx = -1
                        closest_result_pair = None
                        for tensor_idx, j, k in result_pairs:
                            # 提取对应的 tensor 中的点
                            point = lane_2d[i][tensor_idx][j, :] #这里的 j 是点的索引，k 可能在这个场景下不用
                            # 计算 x 轴上的距离
                            distance_x = abs(float(point[0]) - float(right_down[0]))
                            # 更新最小距离和对应的序号
                            if distance_x < min_distance:
                                min_distance = distance_x
                                closest_idx = tensor_idx  # 这里保存的是 tensor 的索引，如果需要保存整体的序号 (tensor_idx, j, k)，则保存 result_pair
                                closest_result_pair = (tensor_idx, j, k)
                        if closest_idx != -1:
                            tensor_idx, j, k = closest_result_pair
                            y1 = lane_2d[i][tensor_idx][j][1]
                            y2 = lane_2d[i][tensor_idx][k][1]
                            x1 = lane_2d[i][tensor_idx][j][0]
                            x2 = lane_2d[i][tensor_idx][k][0]
                            box_middle_x = (box[0] + box[2]) / 2.0
                            y_3d = (right_down[1] - y1) / (y2 - y1) * (lane_3d[i][k][1] - lane_3d[i][j][1]) + lane_3d[i][j][1]
                            x_3d = (box_middle_x - x1) / (x2 - x1) * (lane_3d[i][k][0] - lane_3d[i][j][0]) + lane_3d[i][j][0]
                            proposal_boxes_3d.append((box_id, x_3d, y_3d))

    def _bbox_forward_train(self, x, proposal_list, img_metas):
        """Run forward function and calculate loss for box head in training."""

        bbox_results = self._bbox_forward(x, proposal_list, img_metas)
        bbox_results.update(pred={'cls_scores': bbox_results['cls_scores'], 'bbox_preds': bbox_results['bbox_preds']})

        return bbox_results
    
    def _bbox_forward(self, x, proposal_list, img_metas):
        time_stamp = np.array([img_meta['timestamp'] for img_meta in img_metas])#12个时间戳
        mean_time_stamp = time_stamp[self.num_views:].mean() - time_stamp[:self.num_views].mean()#前一帧的6张图片的时间戳均值-当前帧的6张图片时间戳均值,约为0.5

        bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas)

        if len(img_metas) > self.num_views:#12>6
            bbox_preds = bbox_results['bbox_preds']#预测box
            bbox_preds_with_time = [
                torch.cat([pred[..., :8], pred[..., 8:] / mean_time_stamp], dim=-1) for pred in bbox_preds]#这是在干啥？
            bbox_results['bbox_preds'] = bbox_preds_with_time#把处理过的新预测box填充回对应字段

        return bbox_results
    
    def _bbox_forward_denoise(self, x, proposal_list, img_metas):
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:#2D检测框为0的情况下
            proposal = torch.tensor([[0, 50, 50, 100, 100, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]#填充假的

        rois = bbox2roi(proposal_list)#给每个框的编码前加一个所属的图片索引,变成5维的向量，最后把所有的cat一下，变成(n,5)的tensor
        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        # 把不同size的候选框从原图上映射到选中的feature map上，然后转化为设定好的等长向量(譬如7x7),x0[[12, 512, 32, 88]],eg:bbox_feats[40, 512, 7, 7]
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)#特征维度:512
        bbox_feats, _ = bbox_feats.split([c // 2, c // 2], dim=1)#只要前256维度特征，即只要图像特征，不要pe

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        )

        # query generator
        reference_points, return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, extra_feats)
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        reference_points.clamp(min=0, max=1)

        # split image features and 3dpe
        feat, pe = x[self.feat_lvl].split([c // 2, c // 2], dim=1)  # [num_views, c, h, w]
        stride = self.strides[self.feat_lvl]

        # box correlation
        num_rois_per_img = [len(p) for p in proposal_list]
        feat_for_rois = self.box_corr_module.gen_box_correlation(rois, num_rois_per_img, img_metas, feat, stride)

        # generate image padding mask
        num_views, c, h, w = feat.shape
        mask = torch.zeros_like(feat[:, 0]).bool()  # [num_views, h, w]
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape']
        mask_outside = feat.new_ones((1, num_views, input_img_h, input_img_w))
        for img_id in range(num_views):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            mask_outside[0, img_id, :img_h, :img_w] = 0
        mask_outside = F.interpolate(mask_outside, size=feat.shape[-2:]).to(torch.bool)[0]
        mask[mask_outside] = 1

        # generate cross attention mask
        cross_attn_mask = ~feat_for_rois
        if self.training:
            invalid_rois = cross_attn_mask.view(cross_attn_mask.size(0), -1).all(1)
            cross_attn_mask[invalid_rois, 0, 0, 0] = 0

        roi_mask = (~cross_attn_mask).any(dim=0)  # [num_views, h, w], 1 for valid
        feat = feat.permute(0, 2, 3, 1)[roi_mask][..., None, None]  # [num_valid, c, 1, 1]
        pe = pe.permute(0, 2, 3, 1)[roi_mask][..., None, None]
        mask = mask[roi_mask][..., None, None]
        cross_attn_mask = cross_attn_mask[:, roi_mask][..., None, None]  # [num_rois, num_valid, 1, 1]

        # denoise training
        if self.use_denoise:
            num_ori_reference = len(reference_points)
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(
                1, reference_points, img_metas[0:1], num_ori_reference)
            reference_points = reference_points[0]
            num_pad_reference = len(reference_points) - num_ori_reference
            pad_cross_attn_mask = cross_attn_mask.all(dim=0)[None].repeat(num_pad_reference, 1, 1, 1)
            cross_attn_mask = torch.cat([pad_cross_attn_mask, cross_attn_mask], dim=0)
        else:
            attn_mask = None
            mask_dict = None

        all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                        feat[None],
                                                        mask[None],
                                                        pe[None],
                                                        attn_mask=attn_mask,
                                                        cross_attn_mask=cross_attn_mask,
                                                        force_fp32=self.force_fp32, )#all_cls_scores:[6, 1, 120, 10],all_bbox_preds:[6, 1, 120, 10]

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]#[6, 1, 80, 10]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]#[6, 1, 80, 10]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)#mask_dict.keys()=['known_indice', 'batch_idx', 'map_known_indice', 'known_lbs_bboxes', 'know_idx', 'pad_size', 'output_known_lbs_bboxes']
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]#[6, 1, 40, 10]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))

        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        )

        return bbox_results