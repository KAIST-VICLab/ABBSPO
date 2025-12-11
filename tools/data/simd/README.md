# Preparing SIMD Dataset

<!-- [DATASET] -->

```bibtex
@article{haroon2020multisized,
  title={Multisized object detection using spaceborne optical imagery},
  author={Haroon, Muhammad and Shahzad, Muhammad and Fraz, Muhammad Moazam},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={13},
  pages={3032--3046},
  year={2020},
  publisher={IEEE}
}
```

## download simd dataset

The simd dataset can be downloaded from [here](https://github.com/ihians/simd).

The data structure is as follows:

```none
abbspo
├── mmrotate
├── tools
├── configs
├── data
│   ├── SIMD
│   │   ├── train
│   │   ├── val
```

## split simd dataset

Please crop the original images into 768×768 patches with an overlap of 200 by run

```shell
python tools/data/simd/split/img_split.py --base-json \
  tools/data/simd/split/split_configs/ss_train.json

python tools/data/simd/split/img_split.py --base-json \
  tools/data/simd/split/split_configs/ss_val.json
```

If you want to get a multiple scale dataset, you can run the following command.

```shell
python tools/data/simd/split/img_split.py --base-json \
  tools/data/simd/split/split_configs/ms_train.json

python tools/data/simd/split/img_split.py --base-json \
  tools/data/simd/split/split_configs/ms_val.json
```

Please update the `img_dirs` and `ann_dirs` in json.

The new data structure is as follows:

```none
abbspo
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_simd
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val_full
│   │   │   ├── images
│   │   │   ├── annfiles
```

Note: `val_full` stores the original validation images at their original resolution(before patch splitting), 
which can be used for full-image evaluation and visualization.

Please change `data_root` in `configs/_base_/datasets/simd.py` to `data/split_ss_simd`.

## Generating COCO style annotations

Please convert the annotations from txt to json by run

```shell
python tools/data/simd/dota2coco.py \
  data/split_ss_simd/train/ \
  data/split_ss_simd/train.json

python tools/data/simd/dota2coco.py \
  data/split_ss_simd/val/ \
  data/split_ss_simd/val.json

python tools/data/simd/dota2coco.py \
  data/split_ss_simd/val_full/ \
  data/split_ss_simd/val_full.json
```

The new data structure is as follows:

```none
abbspo
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_simd
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   │   ├── train.json
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   │   ├── val.json
│   │   ├── val_full
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   │   ├── val_full.json
```

Please change `data_root` in `configs/_base_/datasets/dota_coco.py` to `data/split_ss_simd`.
