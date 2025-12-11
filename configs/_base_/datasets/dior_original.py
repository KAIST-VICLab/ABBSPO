# dataset settings
dataset_type = 'DIORDataset'
data_root = '/home/kaist/h2rbox-v2/data/DIOR/'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='ImageSets/Main/train.txt',
        ann_file='ImageSets/Main/trainval.txt',
        ann_subdir='Annotations/Horizontal Bounding Boxes/',  ## Weakly Supervised 
        # ann_subdir='Annotations/Oriented Bounding Boxes/',      ## Fully Supervised
        data_prefix=dict(img_path='JPEGImages-trainval'),
        filter_cfg=dict(filter_empty_gt=True),
        ann_type='hbb',                                       ## Weakly Supervised 
        # ann_type='obb',                                         ## Fully Supervised
        pipeline=train_pipeline))
        # datasets=[
        #     dict(
        #         type=dataset_type,
        #         data_root=data_root,
        #         ann_file='ImageSets/Main/train.txt',
        #         data_prefix=dict(img_path='JPEGImages-trainval'),
        #         filter_cfg=dict(filter_empty_gt=True),
        #         pipeline=train_pipeline),
        #     dict(
        #         type=dataset_type,
        #         data_root=data_root,
        #         ann_file='ImageSets/Main/val.txt',
        #         data_prefix=dict(img_path='JPEGImages-trainval'),
        #         filter_cfg=dict(filter_empty_gt=True),
        #         pipeline=train_pipeline,
        #         backend_args=backend_args)
        # ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        # ann_file='ImageSets/Main/test_vis.txt',
        # ann_file='ImageSets/Main/test_airplane.txt',
        # ann_file='ImageSets/Main/test_paper.txt',
        
        # ann_file='ImageSets/Main/test_gt.txt',
        # ann_subdir='Annotations/Merge Bounding Boxes/', 
        data_prefix=dict(img_path='JPEGImages-test'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

# val_evaluator = dict(type='DOTAMetric', metric='mAP')
# val_evaluator = dict(type='OURMetric', metric='mAP', iou_thrs=[0.5, 0.75], predict_box_type='rbox',)
val_evaluator = dict(type='OURMetric', metric='mAP', iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], predict_box_type='rbox', measure_fps=True)
test_evaluator = val_evaluator
