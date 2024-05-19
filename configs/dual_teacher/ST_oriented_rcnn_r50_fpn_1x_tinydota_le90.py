_base_="base.py"

data_root = '/home/wsoffice/disk/datasets/DOTA-tiny/'
save_name = 'debug'
angle_version = 'le90'
fold = 1
percent = 1

work_dir = "./work_dirs/{}/{}/{}".format(save_name,fold,percent)
log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type="DOTADatasetv2",
            ann_file=data_root + "train1024/labelTxt_semi_supervised/labeled.{}@{}/".format(fold,percent),
            img_prefix=data_root + 'train1024/images/',
            version=angle_version,
            #filter_empty_gt=False,
        ),
        unsup=dict(
            type="DOTADatasetv2",
            ann_file=data_root + "train1024/labelTxt_semi_supervised/unlabeled.{}@{}/".format(fold,percent),
            img_prefix=data_root + 'train1024/images/',
            version=angle_version
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)