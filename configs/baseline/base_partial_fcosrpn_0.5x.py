"""
Supervised training on DOTA v1.5, using partial data, i.e., no unsup images involved
1.Use FCOS in RPN
2. use 4 GPUs

/root/miniconda3/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, 
it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)

TODO():
1. use 2 layers in 'stacked_convs'
2. disable centerness
3. adopt 2x
4. disable center sampling
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

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) #lr=0.0025,

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="DOTADataset",
        ann_file=data_root + 'semisup/train/labelTxt_${percent}p_labeled',
        img_prefix=data_root + 'train/images',
        version="${angle_version}",
        classes="${classes}",
    ),

    val=dict(type="DOTADataset",
            classes="${classes}",
            version="${angle_version}", 
            ann_file=data_root + 'val/labelTxt/',
            img_prefix=data_root + 'val/images/',
            ),

    test=dict(type="DOTADatasetv2",
            classes="${classes}",
            version="${angle_version}", 
            ann_file=data_root + 'val/labelTxt/',
            img_prefix=data_root + 'val/images/',
            ),
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
        _delete_=True,  # ignore the unused old settings

        type='RotatedFCOSHead_ST',
        num_classes=1,
        in_channels=256,
        disable_centerness=False,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        center_sampling=True, 
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        version="${angle_version}",
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version="${angle_version}"),
        loss_cls=dict(
            type='TruncatedFocalLoss', #'FocalLoss',
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
                angle_range="${angle_version}",
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

# find_unused_parameters = True
evaluation = dict(interval=4, metric='mAP')
