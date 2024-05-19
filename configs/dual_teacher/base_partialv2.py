"""Test on val set

Test set settting does not support automation on 'class' and 'angle_version'
"""

_base_="base.py"



work_dir = "work_dirs/${cfg_name}/${percent}-2"
data_root = '/root/autodl-tmp/datasets/DOTAv1.5_1024/'
percent = 10
angle_version = 'le90'
#DOTA-v1.5
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane')

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type="DOTADataset",
            ann_file=data_root + 'semisup/train/labelTxt_${percent}p_labeled',
            img_prefix=data_root + 'train/images',
            version="${angle_version}",
            classes="${classes}",
        ),
        unsup=dict(
            type="DOTADataset",
            ann_file=data_root + 'semisup/train/labelTxt_${percent}p_unlabeled',
            img_prefix=data_root + 'train/images',
            version="${angle_version}",
            classes="${classes}",
        ),
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

 
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler2",
            sample_ratio=[1,4],
            epoch_length=5000,
            by_prob=False
        )
    ),
)



log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)



model = dict(
    rpn_head=dict(
        version="${angle_version}",
        ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
            )),
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
    angle_version = "${angle_version}",
)