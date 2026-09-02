# TIDE-ST

## Temporal- and Infection-aware Deep Learning Framework for Spatial Transcriptomics Deconvolution

TIDE-ST is a deep learning framework designed for spatial transcriptomics deconvolution in dynamic host–pathogen interaction systems.

Unlike conventional deconvolution approaches that mainly model static cellular composition, TIDE-ST integrates temporal information, infection-associated supervision, and cell-type-aware optimization to improve cell-type proportion estimation during infection progression.

---

## Overview

Spatial transcriptomics spots often contain mixed signals from multiple cell populations. 
TIDE-ST addresses this challenge by incorporating:

- **Temporal-aware representation learning** to capture infection-stage-dependent changes.
- **Infection-associated auxiliary supervision** to provide biological constraints.
- **Cell-type-aware optimization** to improve reconstruction of diverse cell populations.

The framework was developed and evaluated using rice blast infection spatial transcriptomics datasets.

---

## Installation

Create an environment:

```bash
conda create -n tidest python=3.9
conda activate tidest
Reference profiles
        |
        v
Source-level split
        |
        v
Pseudo-spot generation
        |
        v
Model training and evaluation
