<div align="center">

  <img src="assets/abbspo_logo.png" width="450"/>

  <div>&nbsp;</div>

  <div align="center">
    <b><font size="5">ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior Based Orientation Prediction for Detecting Aerial Image Objects</font></b>
    <br/>
    <b><font size="4">CVPR 2025</font></b>
  </div>

  <div>&nbsp;</div>

  <!-- Optional badges (add later)
  [![arXiv](https://img.shields.io/badge/arXiv-2501.01234-b31b1b.svg)](https://arxiv.org)
  [![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
  -->

  [📄 Paper](<PAPER_LINK_HERE>) |
  [🌐 Project Page](https://kaist-viclab.github.io/ABBSPO_site/) |
  [📦 Checkpoints (TBA)](#-pretrained-models) |
  [🤝 Issues](https://github.com/KAIST-VICLab/ABBSPO/issues)

</div>

---

<div align="center">

**Woojin Lee\***, **Hyugjae Chang\***, Jaeho Moon, Jaehyup Lee†, Munchurl Kim†  
KAIST, KNU  
\*Co-first authors &nbsp;&nbsp; †Co-corresponding authors  

</div>

---

<p align="center">
  <img src="assets/abbspo_teaser.png" width="800"/>
</p>

<p align="center">
ABBSPO is a weakly supervised oriented object detection (WS-OOD) framework built upon H2RBox-v2.
It learns oriented bounding boxes (RBoxes) using only horizontal bounding boxes (HBoxes) via
<b>Adaptive Bounding Box Scaling (ABBS)</b> and a <b>Symmetric Prior Angle (SPA)</b> loss.
</p>

---

# 🗞️ News

- **Feb 2025** — Initial code release.  
- **Jan 2025** — Paper accepted to **CVPR 2025**.  
- **Coming soon** — Checkpoints for DIOR-R and DOTA-v1.0.

---

# ⚙️ Installation

## Option A — Install using pip & requirements.txt
```bash
git clone https://github.com/KAIST-VICLab/ABBSPO.git
cd ABBSPO

conda create -n abbspo python=3.8
conda activate abbspo

pip install -r requirements.txt
pip install -v -e .
