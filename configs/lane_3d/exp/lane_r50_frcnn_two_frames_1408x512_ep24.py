_base_ = [
    '../data/two_frames.py', '../detectors/maskrcnn_r50.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
post_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
roi_size = 7
roi_srides = [16]

model = dict(
    type='LANE3D',
    # use_grid_mask=dict(
    #     use_h=True,
    #     use_w=True,
    #     rotate=1,
    #     offset=False,
    #     ratio_range=(0.4, 0.6),
    #     mode=1,
    #     prob=0.7,
    #     interv_ratio=0.8
    # ),
    # NOTE:
    # the FPN in faster r-cnn starts from p2
    # we use p4 (downsample rate: 16)
    base_detector=dict(
        backbone=dict(
            with_cp=True,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256, 256],
        out_channels=256,
        start_level=2,
        end_level=2,
        num_outs=1,
    ),
    roi_head=dict(
        type='LANE3DHead',
        pc_range=point_cloud_range,
        force_fp32=True,

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),
            featmap_strides=roi_srides,
            out_channels=256, ),
        bbox_head=dict(
            type='LANEBoxHead',
            num_classes=10,
            pc_range=point_cloud_range,
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=post_range,
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10),
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        ),
    ),
    train_cfg=dict(
        complement_2d_gt=0.4,
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),#最小的2dbox像素长度和宽度
        rcnn=dict(
            stage_loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range),
            sampler_cfg=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(nms_thr=1.0, use_rotate_nms=True, ),
            max_per_scene=300,
        ))
)

data = dict(
    workers_per_gpu=0,#debug模式
    train=dict(
        load_separate=True,
    ),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'base_detector.backbone': dict(lr_mult=0.25),
        }
    ),
    weight_decay=0.01)

optimizer_config = dict(
    _delete_=True,
    type='Fp16OptimizerHook', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=3, )
find_unused_parameters = False
log_config = dict(interval=50)



