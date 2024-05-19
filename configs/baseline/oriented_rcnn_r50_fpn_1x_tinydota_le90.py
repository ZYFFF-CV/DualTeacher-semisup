_base_ = "base.py"
fold = 1
percent = 1

data_root = '/home/wsoffice/disk/datasets/DOTA-tiny/'
save_name = 'baseline_tinydota_all'
angle_version = 'le90'
classes = ('plane',  'bridge', 'small-vehicle', 'large-vehicle',
                'ship', 'storage-tank','swimming-pool', 'helicopter')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type="DOTADataset",
        classes=classes,
        ann_file=data_root + 'train1024/labelTxt_tiny/',
        img_prefix=data_root + 'train1024/images/',
        version=angle_version,    
    ),
    val=dict(classes=classes,
            ann_file=data_root + 'val1024/labelTxt_tiny/',
            img_prefix=data_root + 'val1024/images/',),
    test=dict(classes=classes,
            ann_file=data_root + 'test1024/labelTxt_tiny/',
            img_prefix=data_root + 'test1024/images/',),
)



work_dir = "./work_dirs/{}/{}/{}".format(save_name,fold,percent)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)