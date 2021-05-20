_base_ = '/opt/ml/code/mmdetection_trash/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py'
model = dict(
    pretrained=True,
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        model_type='efficientnet-b0',
        out_indices=(3, 4, 5, 6)),
    neck=dict(
        type='FPN',
        in_channels=[112, 192, 320, 1280],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=11)
    ))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
           'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
           'Clothing')
data = dict(
    samples_per_gpu=32,
    train=dict(
        ann_file='/opt/ml/input/data/train.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=train_pipeline,
        classes=classes
    ),
    val=dict(
        ann_file='/opt/ml/input/data/val.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=test_pipeline,
        classes=classes
    ),
    test=dict(
        ann_file='/opt/ml/input/data/test.json',
        img_prefix='/opt/ml/input/data/',
        pipeline=test_pipeline,
        classes=classes
    ),
)
optimizer = dict(_delete_=True, type='Adam', lr=0.0003, weight_decay=0.0001)
evaluation = dict(metric='bbox', interval=1, save_best='bbox_mAP_50')
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-05)
runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(interval=1, max_keep_ckpts=6)
