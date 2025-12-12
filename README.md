<div align="center">
<h2>
[CVPR 2025] ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior Based Orientation Prediction for Detecting Aerial Image Objects
</h2>

<div>
    <b>Woojin Lee*</b>&nbsp;&nbsp;&nbsp;
    <b>Hyugjae Chang*</b>&nbsp;&nbsp;&nbsp;
    Jaeho Moon&nbsp;&nbsp;&nbsp;
    Jaehyup Lee<sup>†</sup>&nbsp;&nbsp;&nbsp;
    Munchurl Kim<sup>†</sup>
</div>

<br>
<div>
    <sup>*</sup>Co-first authors &nbsp;&nbsp;
    <sup>†</sup>Co-corresponding authors
</div>

<div>
    KAIST (Korea Advanced Institute of Science and Technology), KNU
</div>

<br>

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

<h4>
This repository is the official PyTorch implementation of 
<b>"ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior Based Orientation Prediction for Detecting Aerial Image Objects"</b>.
ABBSPO is a weakly supervised oriented object detection (WS-OOD) framework that learns accurate rotated bounding boxes using only horizontal bounding box annotations.
</h4>

---

<p align="center">
  <img src="assets/figure2.png" width="1000"/>
</p>

<p align="center">
  <a href="assets/figure2.pdf">[Download high-resolution PDF]</a>
</p>

---

## 📧 News
- **Mar 2025** — Initial code release  
- **Jan 2025** — Paper accepted to **CVPR 2025**

---

## 🔧 Tested Environment
- OS: Ubuntu 20.04
- Python: 3.8
- PyTorch: 1.13+
- CUDA: 11.3 / 11.7
- GPU: NVIDIA RTX series

---

## ⚙️ Environment Setup

```bash
git clone https://github.com/KAIST-VICLab/ABBSPO.git
cd ABBSPO

conda create -n abbspo python=3.8
conda activate abbspo

pip install -r requirements.txt
pip install -v -e .
