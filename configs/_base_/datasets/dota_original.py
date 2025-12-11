# dataset settings
dataset_type = 'DOTADataset'
data_root = '/home/kaist/h2rbox-v2/data/dota_1024/split_ss_dota'   ## weakly supervised
# data_root ='/home/kaist/h2rbox-v2/data/split_ss_dota'            ## RBox supervised
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
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
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
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
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/kaist/h2rbox-v2/data/dota_1024/split_ss_dota/val/annfiles/',
        data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/split_ss_dota/val/images/'),
        filter_cfg=dict(filter_empty_gt=True),   ## addition 
        # test_mode=True,                        ## Remove 
        pipeline=val_pipeline))
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
        # ann_file='val/annfiles/',
        # data_prefix=dict(img_path='val/images/'),
        
        # data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/test/images/'),
        
        # ann_file='/home/kaist/h2rbox-mmrotate/data/dota_1024/split_ss_dota/val/annfiles/',
        # data_prefix=dict(img_path='/home/kaist/h2rbox-mmrotate/data/dota_1024/split_ss_dota/val/images/'),

        # ann_file='/home/kaist/h2rbox-mmrotate/data/dota_1024/val/labelTxt_oriented/',
        ann_file='/home/kaist/h2rbox-mmrotate/data/dota_1024/val/labelTxt/',
        data_prefix=dict(img_path='/home/kaist/h2rbox-mmrotate/data/dota_1024/val/images/'),

        # ann_file='/home/kaist/h2rbox-mmrotate/data/dota_1024/val_vis/labelTxt_oriented/',
        # data_prefix=dict(img_path='/home/kaist/h2rbox-mmrotate/data/dota_1024/val_vis/images/'),

        # ann_file='/home/kaist/h2rbox-v2/data/dota_1024/val_vis/labelTxt_merge/',
        # data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/val_vis/images/'),
        
        filter_cfg=dict(filter_empty_gt=True),   ## addition 
        # test_mode=True,                        ## Remove 
        pipeline=val_pipeline))

# val_evaluator = dict(type='DOTAMetric', metric='mAP')
val_evaluator = dict(type='OURMetric', metric='mAP')
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/test/images/'),
#         # data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/test_split/images/'),
#         # data_prefix=dict(img_path='/home/kaist/h2rbox-v2/data/dota_1024/test_paper/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/dota/Task1')
