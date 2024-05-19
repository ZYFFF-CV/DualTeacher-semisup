"""
TODO():
1. Load pretrained weights
2. adopt 2x

"""

_base_="base_partial.py"
checkpoint_path = '/root/autodl-tmp/workdirs/noisedet/base_pretrain/epoch_37.pth'

work_dir = "work_dirs/baseline/${cfg_name}/${percent}"
angle_version = 'le90'
data_root = '/root/autodl-tmp/datasets/DOTAv1.5_1024/'

#DOTA-v1.5
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) 

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
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='${checkpoint_path}', 
                      prefix='student.backbone.')
                      ),
    neck=dict(
        
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=None
        ),
)