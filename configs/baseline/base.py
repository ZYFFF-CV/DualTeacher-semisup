"""
compared with original 'base.py' in 'SoftTeacher'
1. 'RandResize' in 'train_pipeline' is not used
2. __base__ is different in 'datasets' and 'models'.
3. 'img_scale' is [1024,1024] in 'MultiScaleFlipAug' of 'test_pipeline'
4. The lr config, max_poch is utilized as in mmrotte, 
rather than using max interations and higher lr as in 'SoftTeacher'
5. use 'RRandFlip' to replace 'RandFlip'

6. Backbone setting in SoftTeacher, which is urrently not used
model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    )
)

as well as :
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
is not used 

7. fp16 = dict(loss_scale="dynamic") is not used
8. use oriented rcnn rather than fasterrcnn as baseline
8. full name should be 'oriented_rcnn_r50_fpn_1x_dotav2_le90_partial'
"""
_base_ = [
    '../_base_/datasets/dotav2.py', 
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',

   
]


data_root = '/home/wsoffice/disk/datasets/DOTA-tiny/'
save_name = 'baseline_oriented_rcnn_r50_dotatiny'
angle_version = 'le90'
classes = ('plane',  'bridge', 'small-vehicle', 'large-vehicle',
                'ship', 'storage-tank','swimming-pool', 'helicopter')
work_dir = "./work_dirs/{}".format(save_name)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(type="RRandFlip", flip_ratio=0.5,version=angle_version),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        angle_version=angle_version,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RRandFlip", version=angle_version),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="DOTADatasetv2",
        classes=classes,  
        img_prefix=data_root + 'train1024/images/',
        ann_file=data_root + 'train1024/labelTxt_tiny/',
        version=angle_version, 
        pipeline=train_pipeline   
    ),

    val=dict(classes=classes, 
            version=angle_version,
            ann_file=data_root + 'val1024/labelTxt_tiny/',
            img_prefix=data_root + 'val1024/images/',
            pipeline=test_pipeline
            ),

    test=dict(classes=classes,
            version=angle_version,
            ann_file=data_root + 'test1024/labelTxt_tiny/',
            img_prefix=data_root + 'test1024/images/',
            pipeline=test_pipeline
            ),
)



log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)


model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))