"""Note that when using transferlearning, i.e. finetune, run 'tools/model_converters/publish_model.py' 
to remove unnecessary info
"""

_base_="base_partial_fcosrpn_0.5x.py"

load_from = '/root/autodl-tmp/workdirs/noisedet/baseline/base_full_sodaa_trunc_fcosrpn_4x/10/published_latest-a2a7bdb2.pth'  # noqa

# optimizer
# lr is set for a batch size of 4gpu*4
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0/3,
    step=[15])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=16)
log_config = dict(interval=10)

# set backbone freezed
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        ),

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