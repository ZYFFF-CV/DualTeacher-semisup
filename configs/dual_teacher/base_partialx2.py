_base_="base_partialv2.py"
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)