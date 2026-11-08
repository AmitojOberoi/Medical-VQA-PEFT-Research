# Medical VQA: Parameter-Efficient Fine-Tuning Research

**Optimizing Parameter-Efficient Fine-Tuning of Vision-Language Models for Medical Visual Question Answering: A Comparative Study Across Radiology and Pathology Modalities**

## Overview

Medical Visual Question Answering (Med-VQA) is a research area that combines computer vision and natural language processing to answer clinical questions about medical images.

Large Vision-Language Models (VLMs) offer promising capabilities for medical image understanding. However, fine-tuning these models can require substantial computational resources.

This project investigates and compares **Parameter-Efficient Fine-Tuning (PEFT)** techniques for medical visual question answering, focusing on model performance, computational efficiency and robustness across medical imaging modalities.

The planned study compares:

- **LoRA (Low-Rank Adaptation)**
- **QLoRA (Quantized Low-Rank Adaptation)**
- **Prompt Tuning**

The research will investigate their effectiveness across radiology and pathology datasets.

## Research Objectives

1. Compare the performance of LoRA, QLoRA and Prompt Tuning for medical VQA.
2. Evaluate computational efficiency, including training time, GPU memory consumption and trainable parameters.
3. Investigate model generalization across different medical VQA datasets.
4. Examine robustness across radiology and pathology modalities.
5. Identify trade-offs between prediction quality and computational cost.

## Research Datasets

The project plans to work with the following datasets:

| Dataset | Modality | Purpose |
|---|---|---|
| VQA-RAD | Radiology | Medical visual question answering |
| SLAKE | Radiology and medical images | Medical VQA and cross-dataset evaluation |
| PathVQA | Pathology | Pathology-focused medical VQA |

Dataset access, licenses, preprocessing requirements and evaluation splits will be documented before experimentation.

## Methods Under Investigation

### 1. LoRA

Low-Rank Adaptation introduces trainable low-rank matrices into selected model layers while keeping most pretrained parameters frozen.

### 2. QLoRA

QLoRA combines quantization with LoRA to reduce the memory requirements of fine-tuning large models.

### 3. Prompt Tuning

Prompt Tuning introduces trainable virtual tokens or embeddings while keeping the pretrained model parameters frozen.

The implementations will be adapted to the selected vision-language model architecture.

## Evaluation Strategy

The experimental evaluation will consider four major dimensions.

**Prediction performance**
- Accuracy
- Exact Match
- F1 score

**Computational efficiency**
- Training time
- GPU memory consumption
- Number of trainable parameters
- GPU-hours

**Generalization**
- Cross-dataset evaluation
- Evaluation on different medical VQA datasets

**Robustness**
- Cross-modality evaluation
- Prompt variation sensitivity
- Performance under changes in evaluation conditions

Evaluation protocols will distinguish between open-ended and closed-ended questions where appropriate.

## Project Architecture

```text
Medical-VQA-PEFT-Research/
│
├── configs/
│   ├── config.yaml
│   ├── lora.yaml
│   ├── qlora.yaml
│   └── prompt_tuning.yaml
│
├── data/
│   ├── raw/
│   │   ├── VQA_RAD/
│   │   ├── SLAKE/
│   │   └── PathVQA/
│   ├── processed/
│   └── splits/
│
├── preprocessing/
│   ├── prepare_data.py
│   └── validate_data.py
│
├── train_pipeline.py
├── evaluate_pipeline.py
├── robustness_pipeline.py
├── run_all.py
├── utils.py
│
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   ├── logs/
│   └── graphs/
│
├── tests/
│   ├── test_data.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── analysis/
│   ├── compare_methods.py
│   ├── generate_graphs.py
│   └── statistical_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Technology Stack

The planned technology stack includes:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- BitsAndBytes (for compatible QLoRA environments)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyYAML

The exact model, library versions and GPU configuration will be finalized during environment setup.

## Experimental Workflow

The project follows this workflow:

1. Dataset acquisition and validation.
2. Data cleaning and preprocessing.
3. Base vision-language model integration.
4. Implementation of LoRA, QLoRA and Prompt Tuning.
5. Controlled fine-tuning experiments.
6. Performance and computational evaluation.
7. Cross-dataset and robustness experiments.
8. Statistical analysis and visualization.
9. Research report preparation.

## Reproducibility

The project is designed with reproducibility in mind.

- Centralized YAML configuration.
- Fixed random seeds where applicable.
- Experiment logging.
- Checkpoint management.
- Consistent evaluation protocols.
- Recorded computational resource usage.

## Current Development Status

**Phase 1: Project Foundation**

- [x] Initial repository architecture.
- [x] Python virtual environment.
- [x] Configuration loader.
- [x] Shared utility functions.
- [x] Random seed management.
- [x] Device detection.
- [ ] Dataset validation.
- [ ] Dataset preprocessing.
- [ ] Base VLM integration.
- [ ] LoRA implementation.
- [ ] QLoRA implementation.
- [ ] Prompt Tuning implementation.
- [ ] Evaluation and robustness experiments.

This repository is actively under development. Experimental results and conclusions will be added after the corresponding experiments have been completed.

## Collaboration

This project is being developed collaboratively, with contributions to data engineering, model implementation, experimentation and research analysis.

## Research Goal

The ultimate goal is to provide a systematic comparison of parameter-efficient fine-tuning strategies for medical vision-language models, examining not only predictive performance but also computational cost and generalization across medical imaging modalities.
