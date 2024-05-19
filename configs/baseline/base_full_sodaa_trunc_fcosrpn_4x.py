"""Train on SODA-A dataset
1. 4x epoch, 4 gpu
2. Trucn focal 
3. Resize to 1024 rather than 1200, also in the test pipeline
"""

_base_="base_partial.py"

work_dir = "work_dirs/baseline/${cfg_name}/${percent}"
angle_version = 'le90'
data_root = '/root/autodl-tmp/datasets/SODA-A-800/'

#SODA-A
classes = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool',
               'windmill')

workflow = [('train', 1)]
evaluation = dict(interval=12, metric='mAP', nproc=4) 

# 48 epoch, 4X
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 44])
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) 
runner = dict(type='EpochBasedRunner', max_epochs=48)


# Dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)), 
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type="SODAADataset",
        classes=classes,
        ann_file=data_root + "train/Annotations/",
        img_prefix=data_root + 'train/Images/',
        angle_version="${angle_version}", 
        pipeline=train_pipeline   
    ),

    val=dict(
        _delete_=True,
        type="SODAADataset",
        classes=classes, 
        angle_version="${angle_version}",
        img_prefix=data_root + 'val/Images/',
        ann_file=data_root + "val/Annotations/",
        pipeline=test_pipeline
        ),

    # Use val in test
    test=dict(
        _delete_=True,
        type="SODAADataset",
        classes=classes,
        angle_version="${angle_version}",
        img_prefix=data_root + 'val/Images/',
        ann_file=data_root + "val/Annotations/",
        pipeline=test_pipeline
        ),
)


model = dict(
    
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='RotatedFCOSHead_ST',
        num_classes=1,
        in_channels=256,
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
        disable_centerness=True,
        stacked_convs=1,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        center_sampling=False, 
        center_sample_radius=3,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        version="${angle_version}",
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version="${angle_version}"),
        loss_cls=dict(
            type='TruncatedFocalLoss', 
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),

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
)