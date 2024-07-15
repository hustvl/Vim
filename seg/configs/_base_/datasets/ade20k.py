# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/mnt/bn/wanqiangdata-cn/dataset/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# load image and output them in a same resolution
# target_size = 640
# benchmark_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', img_scale=(target_size * 4, target_size), keep_ratio=True),
#     # Crop
#     dict(type='RandomCrop', crop_size=(target_size, target_size)),
#     # RANDOM FLIP
#     dict(type='RandomFlip', flip_ratio=0.5),
#     # NORMALIZE
#     dict(type='Normalize', **img_norm_cfg),
#     # IMAGE TO TENSOR
#     dict(type='ImageToTensor', keys=['img']),
#     # COLLECT
#     dict(type='Collect', keys=['img']),
# ]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    # benchmark=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/training',
    #     ann_dir='annotations/training',
    #     pipeline=benchmark_pipeline)        
)
