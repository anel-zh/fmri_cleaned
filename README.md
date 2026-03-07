# fMRI Pain Prediction Pipeline

A modular MATLAB pipeline for predicting continuous pain ratings from fMRI data using activation and dynamic connectivity features.

This repository demonstrates a complete neuroimaging machine-learning workflow including behavioral preprocessing, feature extraction, dataset assembly, predictive modeling, and visualization.

---

## Pipeline Overview

The pipeline consists of five stages:

1. Behavioral preprocessing  
   Continuous ratings → TR-aligned time series → behavioral bins

2. Feature extraction  
   HRF activation (voxel level) and dynamic connectivity (DCC)

3. Dataset assembly  
   Feature alignment, binning, normalization, and model dataset construction

4. Model training  
   Predictive models trained using cross-validation and PC selection

5. Visualization  
   Model comparison and prediction plots

---

## Repository Structure
fmri-pain-prediction-pipeline
│
├── config
│   └── PipelineConfig.m
│
├── processors
│   └── BehavioralProcessor.m
│
├── features
│   └── FMRIFeatureExtractor.m
│
├── data
│   └── DataAssembler.m
│
├── modeling
│   └── ModelTrainer.m
│
├── visualization
│   └── ResultVisualizer.m
│
├── scripts
│   ├── S1_prepare_behavior.m
│   ├── S2_extract_features.m
│   ├── S3_prepare_datasets.m
│   ├── S4_train_models.m
│   └── S5_visualize_results.m

## Dependencies

Required:

MATLAB (tested with R2022+)

Optional:

CANlab / cocoanCORE visualization utilities for advanced plots

The core pipeline runs without these dependencies.

## Running the Pipeline

Run the scripts sequentially:

S1_prepare_behavior.m
S2_extract_features.m
S3_prepare_datasets.m
S4_train_models.m
S5_visualize_results.m

Each script initializes the pipeline configuration and executes a stage of the workflow.

## Output

Running the full pipeline produces:

processed_behavioral_public_example.mat
prepared_data_public_example.mat
results_structure_public_example.mat
validation_summary_public_example.mat
testing_summary_public_example.mat
validation_summary_public_example.csv
testing_summary_public_example.csv

These outputs contain behavioral targets, feature matrices, trained model parameters, and performance summaries.

Author: Anel Zhunussova
