# PyTorch Code for DRAX

This repository contains the PyTorch implementation for DRAX. The code has been tested on Python 3.8 and PyTorch 1.13.0.

## Installation

1. Install the modified Distiller for MVCNN: Place it in the following directory: `/distiller-pytorch-1.13_mvcnn`.

## Dataset Preparation

1. Download the test dataset from [here](https://drive.google.com/file/d/1lBNJXqq3JLfc0oEWDmdA-mrG6NSD2ui6/view?usp=drive_link) and place it under the `/modelnet_test` directory.

## Model Preparation

1. Place the approximate models, which can be downloaded from [here](https://drive.google.com/drive/folders/1htvM7Z5_GYyl0pIb3khws0D0kF54H6UH?usp=sharing), inside the `/Approx_models` directory.
2. Place the accurate models, which can be downloaded from [here](https://drive.google.com/drive/folders/1rTIISm-MWGhYMmxVwpsB7mKdf3Q610hf?usp=sharing), inside the `/acc_models` directory.

## Error Mask Preparation

1. Place the error masks inside the `/error_mask` directory.

## Running DRAX Inference

1. Run the `DRAX_Quality.ipynb` notebook for DRAX Inference.

## DSE Heuristic Threshold

In the context of our system, the Threshold (*Th*) value for the Design Space Exploration (DSE) heuristic has been empirically set to 0.01%.

### Note:
The chosen value for *Th* is based on empirical observations and considerations specific to our system. Adjustments to this threshold may be necessary based on the requirements and characteristics of your application.

## Reference

The implementation is based on the following paper:

- **Towards Energy-Efficient Collaborative Inference using Multi-System Approximations**  
  Authors: Arghadip Das, Soumendu Kumar Ghosh, Arnab Raha, and Vijay Raghunathan  
  Status: Submitted to IEEE Internet of Things Journal