"""Supervised training on DOTA v1.5

train on 8 gpus with strong augmentation

"""

_base_="base_partial.py"

work_dir = "work_dirs/baseline/${cfg_name}/${percent}"
angle_version = 'le90'
data_root = '/root/autodl-tmp/datasets/DOTAv1.5_1024/'

#DOTA-v1.5
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

workflow = [('train', 1)]
evaluation = dict(interval=2, metric='mAP') 

# 24 epoch, 2X
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) 
runner = dict(type='EpochBasedRunner', max_epochs=24)


# Dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)), #(1200, 1200)
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type="DOTADataset",
        classes=classes,
        ann_file=data_root + "train/labelTxt/",
        img_prefix=data_root + 'train/images/',
        version="${angle_version}", 
        pipeline=train_pipeline   
    ),

    val=dict(
        _delete_=True,
        type="DOTADataset",
        classes=classes, 
        version="${angle_version}",
        img_prefix=data_root + 'val/images/',
        ann_file=data_root + "val/labelTxt/",
        pipeline=test_pipeline
        ),

    # Use val in test
    test=dict(
        _delete_=True,
        type="DOTADataset",
        classes=classes,
        version="${angle_version}",
        img_prefix=data_root + 'val/images/',
        ann_file=data_root + "val/labelTxt/",
        pipeline=test_pipeline
        ),
)


model = dict(
    _delete_=True,
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead_plain',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version="${angle_version}"),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))