_base_ = [
    '/opt/ml/code/mmdetection_trash/configs/vfnet/vfnet_r50_fpn_1x_coco.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='RWightman',
        model_name='ecaresnet269d',
        features_only=True,
        pretrained=True,
        out_indices=[1, 2, 3, 4]),
    neck=[dict(
        type='FPN_CARAFE',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=1,
            end_level=-1,
            norm_cfg=None,
            act_cfg=None,
            order=('conv', 'norm', 'act'),
            upsample_cfg=dict(
                type='carafe',
                up_kernel=5,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64)),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
            ])

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00005)
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(max_keep_ckpts=1)
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=0.1,
#    step=[8, 11])
lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    periods=[3, 6, 21, 30],
    restart_weights=[1, 1, 1, 1],
    min_lr=0.0001)


albu_train_transforms = [
    dict(
        type='RandomSizedCrop', min_max_height=[360, 360], height=512, width=512,
        p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        pipeline=train_pipeline))
