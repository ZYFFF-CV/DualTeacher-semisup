_base_="base_partial_soda.py"
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)



data = dict(
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler2",
            sample_ratio=[1,4],
            epoch_length=4000, 
            by_prob=False
        )
    ),
)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) 