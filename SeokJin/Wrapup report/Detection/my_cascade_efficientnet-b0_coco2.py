_base_ = '/opt/ml/code/mmdetection_trash/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained=True,
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        model_type='efficientnet-b0',
        out_indices=(0, 1, 3, 6)),
        #out_indices=(2, 3, 4, 5, 6)),
    neck=dict(
        type='BiFPN',
        in_channels=[24, 40, 112, 1280],
        #in_channels=[80, 112, 192, 320, 1280],
        out_channels=256,
        strides=[4, 8, 16, 32],
        #strides=[4, 8, 16, 32, 64],
        start_level=1,
        #add_extra_convs='on_input',
        #norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ])
)
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
    samples_per_gpu=8,
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
optimizer = dict(_delete_=True, type='SGD', momentum=0.9,
                 lr=0.16, weight_decay=0.00004)
evaluation = dict(metric='bbox', interval=1, save_best='bbox_mAP_50')
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.00001,
    warmup_by_epoch=True,
    min_lr=0
)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
