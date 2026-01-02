<div align="center">
<h2>[CVPR 2025] ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects
</h2>

<div>
    <a href="https://woojin52.github.io/" target="_blank">Woojin Lee</b></a><sup>1*</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://www.viclab.kaist.ac.kr/" target="_blank">Hyugjae Chang</b></a><sup>1*</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://sites.google.com/view/jaehomoon/" target="_blank">Jaeho Moon</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://sites.google.com/view/knuairlab/" target="_blank">Jaehyup Lee</a><sup>2†</sup>&nbsp;&nbsp;&nbsp;
    <a href="https://www.viclab.kaist.ac.kr/" target="_blank">Munchurl Kim</a><sup>1†</sup>
</div>

<br>

<div>
    <sup>*</sup>Co-first authors &nbsp;&nbsp;
    <sup>†</sup>Co-corresponding authors
</div>

<div>
    <sup>1</sup>KAIST (Korea Advanced Institute of Science and Technology), South Korea<br>
    <sup>2</sup>KNU (Kyungpook National University), South Korea
</div>


<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/ABBSPO_site/" target="_blank">
        <img src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.pdf" target="_blank">
        <img src="https://img.shields.io/badge/Paper-CVPR%202025-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/ABBSPO">
    </h4>
</div>
</div>

---

<h4 align="center">
This repository is the official PyTorch implementation of 
<b>"ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects"</b>.
ABBSPO is a weakly supervised oriented object detection (WS-OOD) framework that 
<strong>systematically identifies and addresses the scale and angle supervision mismatch</strong> 
arising from the use of tight horizontal bounding box annotations and 
minimum circumscribed rectangle operations for rotated box generation.
</h4>

---

## 📧 News
- **Dec 15, 2025:** Initial code release  
- **Feb 27, 2025:** Paper accepted to CVPR 2025  

---

## 🔧 Tested Environment
- OS: Ubuntu 22.04
- Python: 3.8
- PyTorch: 1.13.1
- CUDA: 11.7
- GPU: NVIDIA RTX 3090 

## ⚙️ Environment Setup

### Install Environment (Recommended)
```bash
git clone https://github.com/KAIST-VICLab/ABBSPO.git
cd ABBSPO

conda create -n abbspo python=3.8
conda activate abbspo
```

### Step 1. Install PyTorch (CUDA 11.7)
```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Step 2. Install MMCV with CUDA extensions (IMPORTANT)
```bash
pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```
⚠️ Important Note: 
Installing mmcv without the OpenMMLab wheel URL will result in a CPU-only build
and cause runtime errors such as:
```bash
ModuleNotFoundError: No module named 'mmcv._ext'
```

### Step 3. Install remaining dependencies
```bash
pip install -r requirements.txt
pip install -v -e .
```

### (Optional) Installation Sanity Check
```bash
python - <<EOF
from mmcv.ops import batched_nms
print("MMCV CUDA ops are correctly installed.")
EOF
```

## 📁 Data Preparation

Please refer to [tools/data/README.md](tools/data/README.md) for dataset preparation.


## 📦 Pretrained Models
All checkpoints are trained for 12 epochs using the configuration files provided in `configs/abbspo/`.

**Note**:
The DOTA-v1.0 checkpoint is trained **using only the training split** (without validation data).

| Dataset | Model | Training Log |
|--------|-------|--------------|
| DIOR | [model](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_dior_epoch12.pth) | [log](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_dior_epoch12.log) |
| DOTA-v1.0 | [model](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_dota_epoch12.pth) | [log](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_dota_epoch12.log) |
| SIMD | [model](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_simd_epoch12.pth) | [log](https://github.com/KAIST-VICLab/ABBSPO/releases/download/v0.1/abbspo_simd_epoch12.log) |


## 🚀 Get Started

## Training

ABBSPO follows the standard **MMRotate** training pipeline.

Example training command on **DIOR**:

```sh
python tools/train.py \
configs/abbspo/abbspo-le90_r50_fpn-1x_dior.py
```

To train on other datasets, simply change the configuration file:

**DOTA-v1.0**: configs/abbspo/abbspo-le90_r50_fpn-1x_dota.py

**SIMD**: configs/abbspo/abbspo-le90_r50_fpn-1x_simd.py

## Testing 
Example evaluation command on **DIOR**:

```sh
python tools/test.py \
configs/abbspo/abbspo-le90_r50_fpn-1x_dior.py \
work_dirs/abbspo-le90_r50_fpn-1x_dior/epoch_12.pth 
```
To visualize detection results:
```sh
python tools/test.py \
configs/abbspo/abbspo-le90_r50_fpn-1x_dior.py \
work_dirs/abbspo-le90_r50_fpn-1x_dior/epoch_12.pth \
--show-dir visual_results/abbspo-le90_r50_fpn-1x_dior \
--show-score-thr 0.3
```

For more detailed configuration options and advanced usage, please refer to the [MMRotate User Guide](https://mmrotate.readthedocs.io/en/1.x/user_guides/index.html)

## Project Page
Please visit our [project page](https://kaist-viclab.github.io/ABBSPO_site/) for more experimental results.

## Citation
If the content is useful, please cite our paper:
```bibtex
@inproceedings{lee2025abbspo,
  title={ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects},
  author={Lee, Woojin and Chang, Hyugjae and Moon, Jaeho and Lee, Jaehyup and Kim, Munchurl},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8848--8858},
  year={2025}
}
```

## Acknowledgement
This repository is built upon [FMA-Net](https://github.com/KAIST-VICLab/FMA-Net/), [One Look is Enough](https://github.com/KAIST-VICLab/One-Look-is-Enough), and [MMRotate](https://github.com/open-mmlab/mmrotate).
We gratefully thank the [MMRotate](https://github.com/open-mmlab/mmrotate) team and [H2RBox-v2](https://github.com/VisionXLab/point2rbox-mmrotate/tree/dev-1.x/configs/h2rbox_v2) authors 
for their excellent open-source contributions, which made our implementation and experiments much easier.
