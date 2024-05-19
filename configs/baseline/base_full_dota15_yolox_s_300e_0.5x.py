"""Supervised training on DOTA v1.5

train on 4 gpus, double the batchsize than default

Note:
1. 'RMosaic' does not work well and is replaced by 'Mosaic' in MMdet 2.x,
see https://github.com/open-mmlab/mmrotate/issues/469#issuecomment-1227983911 
"""

_base_="base_partial.py"

work_dir = "work_dirs/baseline/${cfg_name}/100"
angle_version = 'le90'
data_root = '/root/autodl-tmp/datasets/DOTAv1.5_1024/'

#DOTA-v1.5
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

workflow = [('train', 1)]


# 300 epoch, 1X

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01 * 8 * 4/ 64, 
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX', 
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='SyncNormHook', 
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook', 
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval,max_keep_ckpts=10)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP')
log_config = dict(interval=50)


# Dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RRandFlip", version="${angle_version}"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


train_dataset=dict(
        _delete_=True,
        type="DOTADataset",
        classes=classes,
        ann_file=data_root + "train/labelTxt/",
        img_prefix=data_root + 'train/images/',
        version="${angle_version}", 
        pipeline=train_pipeline   
    )

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=train_dataset,

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


# model settings
model = dict(
    _delete_=True,
    type='RotatedYOLOX', 
    input_size=img_scale,
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5), 
    neck=dict(
        type='YOLOXPAFPN', 
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='RotatedYOLOXHead', 
        num_classes=len(classes),
        in_channels=128,
        feat_channels=128,
        separate_angle=False,
        with_angle_l1=True,
        angle_norm_factor=5,
        edge_swap="${angle_version}",
        loss_bbox=dict(
            type='RotatedIoULoss', 
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type='RSimOTAAssigner', center_radius=2.5)), 
    test_cfg=dict(
        score_thr=0.01, nms=dict(type='nms_rotated', iou_threshold=0.10)))
