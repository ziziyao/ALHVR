# Adaptive Learning of High-Value Regions for Semi-Supervised Medical Image Segmentation

by Tao Lei, Ziyao Yang, Xingwu Wang, Yi Wang, Xuan Wang, Feiman Sun, Asoke K. Nandi

# Introduction

Official code for "Adaptive Learning of High-Value Regions for Semi-Supervised Medical Image Segmentation".

# Requirements

This repository is based on PyTorch 1.7.0, CUDA 12.2 and Python 3.6.5. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

# Dataset

Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [Brats](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019).

# Training

python train_ALHVR_acdc.py

python train_ALHVR_brats.py

# Testing

python test_acdc.py

python test_brats.py

# Acknowledgements
Our code is adapted from CCT, MC-Net, UPCoL and SSL4MIS. Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.
