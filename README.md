## Introduction

Functional MRI (fMRI) produces high-dimensional, 4D datasets in which neural signals evolve across both space and time. Transforming these thousands of spatial units and temporal acquisition points into structured inputs for predictive modeling presents significant computational challenges. This is further complicated by the need to precisely align continuous behavioral data with imaging acquisition sequences.

This repository provides a modular MATLAB pipeline designed for brain–behavior prediction using time-resolved neuroimaging data. The workflow standardizes the transition from raw data to predictive insights through reproducible stages: behavioral preprocessing, temporal alignment, denoising, neural feature extraction, dataset assembly, and cross-validated modeling.

## Project Motivation

A core objective of this pipeline is to bridge the gap between two traditionally separate analytical approaches: **activation-based** and **connectivity-based** modeling. While most neuroimaging workflows focus on one or the other, this framework integrates both within a unified predictive architecture.

The pipeline extracts and models two complementary feature families:

- **Activation features** derived via single-trial **Hemodynamic Response Function (HRF) regression** at the voxel or atlas-defined region level, capturing localized stimulus-evoked neural responses.
- **Connectivity features** computed using **Dynamic Conditional Correlation (DCC)**, a GARCH-based multivariate time-series method that estimates **time-varying functional interactions between brain regions**.

By combining these representations, the framework captures both **localized neural processing** and **large-scale network dynamics**. This dual-stream modeling approach enables direct comparison of each feature family’s predictive contribution while reducing multicollinearity and improving the interpretability of brain–behavior relationships.

## Repository Note

This public repository features a **demonstration version** of the pipeline using a sustained-pain fMRI case. While simplified to protect unpublished research findings, it preserves the core modular architecture and computational principles of the original pipeline. It serves as a scalable and reproducible template that can be adapted for diverse task-based or resting-state fMRI studies.
