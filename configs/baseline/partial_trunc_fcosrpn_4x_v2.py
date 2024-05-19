"""
1. neck.numouts=5
2. center_sampling=False
3.disable_centerness=True
4.stacked_convs=1
"""

_base_="base_partial_fcosrpn_0.5x.py"

# 48 epoch, 4X
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)

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

        
)