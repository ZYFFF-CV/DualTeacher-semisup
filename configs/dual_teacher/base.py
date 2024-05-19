"""Update
compared with original 'base.py' in 'SoftTeacher'
1. __base__ is different in 'datasets' and 'models'.
2. 'RandResize' in 'train_pipeline', 'strong_pipeline', is not used
3. 'img_scale' is [1024,1024] in 'MultiScaleFlipAug' of 'test_pipeline'
4. use 'DOTADataset' in 'data', instead of 'CocoDataset'
4. The lr config, max_poch is utilized as in mmrotte, 
rather than using max interations and higher lr as in 'SoftTeacher'
5. "SubModulesDistEvalHook" 'interval' is assigned by 1
6. Add 'angle_version' args in the 'SoftTeacher'
7. wandb for visualization is not used
8. use 'RRandFlip' to replace 'RandFlip'
9. RandTranslate is not used in strong_pipeline
10. 'NumClassCheckHook' is not used as there is a bug in there that mmrotate's rpn head is not
    recognised, rising 'num_class unmatched error'
11. '"SemiBalanceSampler"' is replaced by "SemiBalanceSampler2"

12. 'RandResize' in 'weak_pipeline' is modified
13. Add 'angle_range' argument in 'semi_wrapper.train_cfg'
14. 'Resize' in date.test is modified to 'RResize' in mmrotate


9.Backbone setting in SoftTeacher is urrently not used
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

as well as 
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

10. fp16 = dict(loss_scale="dynamic") is not used

Note that:
1. 'SemiBalanceSampler' is defined in the config, it actually calls 
'DistributedGroupSemiBalanceSampler' by adding 'distributed' and 'group' in as prefix, see fucntion 'build_sampler' in 
'ssoddota/datasets/builder.py'
2. The reason of epoch_length = 7330: The number is the epoch length if you run a supervised
 model on the coco dataset with 16 Image/Batch.
 see: https://github.com/microsoft/SoftTeacher/issues/29
 in my case it should be 7330*16/batchsize

"""
_base_ = [
    '../_base_/datasets/dotav2.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
   
]

data_root = '/home/wsoffice/disk/datasets/DOTA-tiny/'
save_name = 'ST_base_partial'
classes = ('plane',  'bridge', 'small-vehicle', 'large-vehicle',
                'ship', 'storage-tank','swimming-pool', 'helicopter')


angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(type="RRandFlip", flip_ratio=0.5),
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
        record=True,
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

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(type="RRandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
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
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
        angle_version=angle_version
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
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
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1024,1024)],
                multiscale_mode='value',
            ),
            dict(type="RRandFlip", flip_ratio=0.5),
        ],
        record=True,
        angle_version=angle_version,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
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
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type="RResize"), 
            dict(type="RRandFlip",version=angle_version),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=None,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="DOTADatasetv2",
            classes=classes,
            ann_file=None,
            img_prefix=None,
            pipeline=train_pipeline,
            version=angle_version,
        ),
        unsup=dict(
            type="DOTADatasetv2",
            classes=classes,
            ann_file=None,
            img_prefix=None,
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
            version=angle_version
        ),
    ),
    val=dict(type="DOTADataset",
            classes=classes,
            pipeline=test_pipeline,
            version=angle_version, 
            ann_file=data_root + 'val1024/labelTxt_tiny/',
            img_prefix=data_root + 'val1024/images/',
            ),
    test=dict(type="DOTADataset",
            classes=classes,
            pipeline=test_pipeline,
            version=angle_version, 
            ann_file=data_root + 'test1024/labelTxt_tiny/',
            img_prefix=data_root + 'test1024/images/',
            ),

    sampler=dict(
        train=dict(
            type="SemiBalanceSampler2",
            sample_ratio=[1, 4],
            by_prob=True,
            epoch_length=23456,
        )
    ),
)

custom_hooks = [
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
evaluation = dict(type="SubModulesDistEvalHook", interval=1)




log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
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
            max_per_img=2000))
    )



semi_wrapper = dict(
    type="SoftTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        angle_range=0,
    ),
    test_cfg=dict(inference_on="student"),
    angle_version = angle_version,
)