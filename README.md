# Generative Modelling Case Study – GANs for Synthetic Data Generation

## Overview

This repository contains the implementation for the **Generative Modelling Case Study** completed as part of the **Advanced Research Topics in Data Science** module.

The project explores the implementation of **Generative Adversarial Networks (GANs)** from scratch and applies them to three real-world domains:

- Medical image synthesis using OCTMNIST retinal images
- Synthetic cybersecurity traffic generation using CICIDS 2017
- Creative sketch generation using Google's QuickDraw dataset

The project demonstrates how different GAN architectures can be adapted to different data types, including low-dimensional distributions, images, and tabular data.

---

## Assignment Objectives

The project consists of two main parts.

### Part 1 – Building GANs from Scratch

- Implement a baseline GAN for a noisy sine-wave distribution
- Generate a noisy spiral distribution
- Modify the GAN architecture
- Compare the original and modified architectures
- Analyse training stability and generated samples

---

### Part 2 – Real-World GAN Applications

#### OCTMNIST

- Explore the OCTMNIST dataset
- Build a Deep Convolutional GAN (DCGAN)
- Generate synthetic retinal images
- Compare real and generated images
- Evaluate using Fréchet Inception Distance (FID)

#### CICIDS 2017

- Load and preprocess Wednesday traffic data
- Train a fully connected GAN
- Generate synthetic network traffic
- Compare distributions using PCA
- Analyse correlation similarity

#### QuickDraw

- Download the Birthday Cake category
- Train a DCGAN
- Generate synthetic sketches
- Compare real and generated images
- Evaluate using FID

---

## Datasets

### 1. OCTMNIST

https://medmnist.com/

---

### 2. CICIDS 2017

https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

---

### 3. Google QuickDraw

https://github.com/googlecreativelab/quickdraw-dataset

---

## Libraries

- Python 3
- PyTorch
- Torchvision
- MedMNIST
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TorchMetrics

---

## Evaluation Metrics

Different evaluation techniques were used depending on the dataset.

### Part 1

- Nearest Neighbour Distance

### OCTMNIST

- Fréchet Inception Distance (FID)
- Visual comparison

### CICIDS

- PCA projection
- Correlation comparison
- Feature statistics

### QuickDraw

- Fréchet Inception Distance (FID)
- Visual comparison

---

## Results

The experiments demonstrate that:

- GANs can successfully learn simple two-dimensional distributions.
- Increasing network depth and batch normalisation improves generation quality for complex distributions.
- DCGANs generate realistic OCT retinal images.
- Tabular GANs capture the overall distribution of CICIDS network traffic.
- DCGANs successfully learn the structure of QuickDraw birthday-cake sketches.

---

## References

- Goodfellow et al. (2014)
- Radford et al. (2016)
- Kingma & Ba (2015)
- Yang et al. (2023)
- Sharafaldin et al. (2018)
- Ha & Eck (2018)
