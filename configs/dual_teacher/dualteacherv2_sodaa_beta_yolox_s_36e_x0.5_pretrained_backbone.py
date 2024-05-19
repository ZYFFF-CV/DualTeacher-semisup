# Use YOLOX-s as the student model and the senior model
# Curently RMosaic is not used, so only 200 epoch
# Use pretrained "cspnext", need download mmcls 1.x
# Use 12 epoch and 4 GPU


_base_="base_partial_soda_0.5x.py"

num_gpus=4


warmup_sup_training_epochs = 6 #6is adequate for soda-a, but need 40 for dota
#15#40 #15 #15 is inadequate #30 # set -1 for debug, DualTeacher, 5 for RFCOS
max_epochs = 12 + warmup_sup_training_epochs #90#160 #300 #36
num_last_epochs = 3#5 #15
interval = 10
resume_from = None #"/root/autodl-tmp/workdirs/noisedet/dualteacherv2_beta_yolox_s_36e_x0.5_pretrained_backbone/30-2/epoch_41.pth"# None
find_unused_parameters=True # only for training 30%

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa
# SODAA softteacher
supervisor_checkpoint='/root/autodl-tmp/base_partial_soda_published_latest-8d4708de.pth'
#SODA-A
classes = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool',
               'windmill')
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0
# Strides of multi-scale prior box
strides = [8, 16, 32]
norm_cfg = dict(type='BN')  # Normalization config
img_scale=(800, 800)

data = dict(
    sampler=dict(
        train=dict(
            type='SemiBalanceSampler2',
            sample_ratio=[1, 4],
            by_prob=False,
            epoch_length=8000//num_gpus)))

evaluation = dict(interval=1, metric='mAP', type='SubModulesDistEvalHook')
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=1)

custom_hooks = [
    dict(
        type='SubModulesSyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type="SubModulesExpMomentumEMAHook", 
        resume_from=resume_from,
        momentum=0.0001,
        priority=49),
    # Semi-sup
    dict(type="WeightSummary"),
    dict(type="SetEpochInfoHook"), 
    dict(type="TeacherWeightUpdater_onestage",reinit_parts=None, upgrade_norm=True), 
     dict(type="MeanTeacher_DualTeacehr_Buffer",
         momentum=0.999, 
         interval=1, 
         upgrade_norm=True) 
]


model_teacher1 = dict(
    init_cfg=dict(type='Pretrained', 
                  checkpoint=supervisor_checkpoint,
                  prefix="student.",
                  ),

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
        ), 
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version="${angle_version}",
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range="${angle_version}",
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)
        ),
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
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )),

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


model_teacher2 = dict(
   
    type='RotatedYOLOX', 
    input_size=img_scale,
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)), 
     neck=dict(
        type='YOLOXPAFPN', 
        in_channels=[256, 512, 1024],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='RotatedYOLOXHead_ST', 
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



model_student = dict(
   
    type='RotatedYOLOX', 
    input_size=img_scale,
    random_size_range=(25, 35),
    random_size_interval=10,
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)), 
     neck=dict(
        type='YOLOXPAFPN', 
        in_channels=[256, 512, 1024],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='RotatedYOLOXHead_ST', 
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

semi_wrapper = dict(
    type="DualTeacherv2_beta",
    _delete_=True,
    model_teacher1="${model_teacher1}",
    model_teacher2="${model_teacher2}",
    model_student="${model_student}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        angle_range=0,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        unsup_cls_loss_type='soft',
        softlearning_after_epoch=warmup_sup_training_epochs,
        student_cls_pos_thr=0.3
    ),
    test_cfg=dict(inference_on="student",
                dual_teacher_score_thr=0.05,
                dual_teacher_nms=dict(iou_thr=0.5),),
    angle_version = "${angle_version}",
)