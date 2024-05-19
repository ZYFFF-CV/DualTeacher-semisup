_base_ = [
    '../_base_/datasets/dotav2.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    #'../_base_/models/oriented_rcnn_r50_fpn_dota_le90.py'
   
]

fold = 1
percent = 10

data_root = '/home/wsoffice/disk/datasets/DOTA-tiny/'
save_name = 'baseline_retinanet_r18_dotatiny_partial'
angle_version = 'le90'
classes = ('plane',  'bridge', 'small-vehicle', 'large-vehicle',
                'ship', 'storage-tank','swimming-pool', 'helicopter')
work_dir = "./work_dirs/{}/{}/{}".format(save_name,fold,percent)

evaluation = dict(interval=6, metric='mAP')
checkpoint_config = dict(interval=4)

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
        # img_scale=(1333, 800),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type="DOTADataset",
        classes=classes,
        ann_file=data_root + "train1024/labelTxt_tiny_semi_supervised/labeled.{}@{}/".format(fold,percent),
        img_prefix=data_root + 'train1024/images/',
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
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4),
    bbox_head=dict(
        type='RotatedRetinaHead',
        # Convert 15 to 8
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))