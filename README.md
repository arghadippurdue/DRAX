# PyTorch Code for DRAX

This repository contains the PyTorch implementation for DRAX. The code has been tested on Python 3.8 and PyTorch 1.13.0.

## Installation

1. Install the modified Distiller for MVCNN: Place it in the following directory: `/distiller-pytorch-1.13_mvcnn`.

## Dataset Preparation

1. Download the test dataset and place it under the `/modelnet_test` directory.

## Model Preparation

1. Place the approximate models inside the `/Approx_models` directory.
2. Place the accurate models inside the `/acc_models` directory.

## Error Mask Preparation

1. Place the error masks inside the `/error_mask` directory.

## Running DRAX Inference

1. Run the `DRAX_Quality.ipynb` notebook for DRAX Inference.

## Reference

The implementation is based on the following paper:

- **Towards Energy-Efficient Collaborative Inference using Multi-System Approximations**  
  Authors: Arghadip Das, Soumendu Kumar Ghosh, Arnab Raha, and Vijay Raghunathan  
  Status: Submitted to IEEE Internet of Things Journal
