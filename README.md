# Medical VQA using Parameter-Efficient Fine-Tuning (PEFT)

A research project exploring **Large Vision-Language Models (LVLMs)** for Medical Visual Question Answering (Medical VQA), using **Low-Rank Adaptation (LoRA)** to adapt a pretrained model to medical image-question-answering datasets.

---

## 1. Project Overview

Medical Visual Question Answering (Medical VQA) is a task in which an AI model receives a medical image and a natural-language question about that image, then generates an appropriate answer.

This project investigates the use of parameter-efficient fine-tuning to adapt a pretrained vision-language model for medical question answering.

Instead of updating all the parameters of a large model, the project uses **LoRA (Low-Rank Adaptation)** to introduce a smaller set of trainable parameters. This reduces the number of parameters that need to be updated during fine-tuning.

### Project Objectives

- Explore the application of Large Vision-Language Models in medical image understanding.
- Prepare and validate multiple Medical VQA datasets.
- Implement a data preprocessing pipeline.
- Apply LoRA for parameter-efficient fine-tuning.
- Train and validate the model on medical question-answering data.
- Evaluate the implementation through automated software tests.
- Compare experimental training and validation losses across datasets.

---

## 2. Model and Architecture

### Base Model

**Model:** `microsoft/llava-med-v1.5-mistral-7b`

**Architecture:** LLaVA with a Mistral 7B language-model backbone.

The model combines visual information from medical images with natural-language questions to generate answers.

### High-Level Workflow

1. A medical image is provided as visual input.
2. A natural-language question is provided alongside the image.
3. The vision encoder processes the image.
4. Visual representations are combined with the language-model input.
5. The language model generates a natural-language answer.

### Parameter-Efficient Fine-Tuning

This project uses **LoRA (Low-Rank Adaptation)**.

LoRA introduces trainable low-rank matrices into selected model layers while keeping the original pretrained weights frozen.

This allows the model to adapt to the target task without updating every parameter in the original model.

---

## 3. LoRA Configuration

The experimental LoRA configuration is:

| Parameter | Value |
|---|---|
| Fine-tuning method | LoRA |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Bias | None |

### Trainable Parameters

| Parameter | Value |
|---|---:|
| Total parameters | 7,570,413,568 |
| Trainable parameters | 4,194,304 |
| Trainable percentage | Approximately 0.0554% |

The small proportion of trainable parameters demonstrates the parameter-efficient nature of the LoRA approach.

---

## 4. Datasets

The project uses three publicly available Medical VQA datasets:

### 4.1 VQA-RAD

VQA-RAD is a Medical VQA dataset containing radiology images and associated questions and answers.

It is used to investigate question answering on radiological images.

### 4.2 SLAKE

SLAKE is a knowledge-enriched Medical VQA dataset containing medical images, questions, and answers.

It includes questions covering different aspects of medical image understanding.

### 4.3 PathVQA

PathVQA is a Medical VQA dataset focused on pathology images. It contains questions and answers related to pathological image content.

It provides an additional dataset for evaluating the application of vision-language models to medical imagery.

### Full Dataset Split Counts

The following table represents the processed dataset counts after preprocessing and validation.

| Dataset | Training | Validation | Testing |
|---|---:|---:|---:|
| VQA-RAD | 1,791 | 216 | 241 |
| SLAKE | 4,918 | 1,053 | 1,061 |
| PathVQA | 19,648 | 6,259 | 6,719 |
| **Total** | **26,357** | **7,528** | **8,021** |

**Note:** These are the full processed dataset counts. The PathVQA training experiment used a reduced subset because of computational constraints. The experimental subset is documented separately below.

---

## 5. Data Preprocessing Pipeline

A preprocessing pipeline was implemented to prepare the datasets for model training.

The pipeline includes:

1. Dataset inspection.
2. Data validation.
3. Image and question-answer record processing.
4. Removal of records with empty answers.
5. Duplicate analysis.
6. Train-validation-test split leakage checks.
7. Conversion into LLaVA-compatible annotation format.
8. Validation of processed annotations and image references.

### Data Cleaning and Validation

The preprocessing pipeline addresses issues such as:

- Empty answers.
- Duplicate question-answer records.
- Potential overlap between dataset splits.
- Invalid or missing image references.
- Incorrect annotation formats.

### LLaVA-Compatible Dataset Preparation

The processed datasets were converted into the annotation format required by the LLaVA training pipeline.

The following records were validated:

| Dataset | Training | Validation | Testing |
|---|---:|---:|---:|
| SLAKE | 4,918 | 1,053 | 1,061 |
| VQA-RAD | 1,791 | 216 | 241 |
| PathVQA | 19,648 | 6,259 | 6,719 |
| **Total** | **26,357** | **7,528** | **8,021** |

The LLaVA-format validation completed successfully, including checks of annotation records and image references.

The split leakage validation also passed.

---

## 6. Experimental Setup

Model fine-tuning experiments were conducted using Kaggle GPU environments with NVIDIA T4 GPUs.

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | LLaVA-Med v1.5 Mistral 7B |
| Fine-tuning method | LoRA |
| Epochs | 1 |
| Learning rate | 0.0002 |
| Training batch size | 1 |
| Evaluation batch size | 1 |
| Gradient accumulation steps | 8 |
| Precision | FP16 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |

### Computational Environment

- GPU: NVIDIA T4
- Training framework: PyTorch
- Model ecosystem: Hugging Face Transformers
- Parameter-efficient fine-tuning: PEFT
- Experiment environment: Kaggle Notebooks

---

## 7. Training Experiments and Results

The following results represent the recorded training and validation losses from the completed experiments.

**Important:** Training loss and validation loss are optimization metrics. They are not equivalent to answer accuracy or clinical correctness.

### 7.1 VQA-RAD

| Metric | Result |
|---|---:|
| Training samples | 1,791 |
| Validation samples | 216 |
| Epochs | 1 |
| Training loss | 0.9294 |
| Validation loss | 0.8054 |
| Training time | 2 hours, 9 minutes, 26 seconds |
| Peak GPU memory | 7.36 GB |

The VQA-RAD LoRA fine-tuning experiment completed successfully, and the trained adapter was backed up.

### 7.2 SLAKE

| Metric | Result |
|---|---:|
| Training samples | 4,918 |
| Validation samples | 1,053 |
| Epochs | 1 |
| Training loss | 0.403944 |
| Validation loss | 0.192734 |
| Peak GPU memory | 7.37 GB |

The SLAKE training experiment completed successfully. The validation run processed all 1,053 validation samples, and the trained adapter was backed up.

### 7.3 PathVQA

Due to computational constraints, the PathVQA experiment used a reduced subset of the full processed dataset.

| Metric | Result |
|---|---:|
| Full processed training split | 19,648 |
| Full processed validation split | 6,259 |
| Actual training subset | 3,916 |
| Actual validation subset | 500 |
| Epochs | 1 |
| Training loss | 0.89736 |
| Validation loss | 0.66694 |
| Training time | 4 hours, 46 minutes, 51 seconds |
| Peak GPU memory | 7.38 GB |

The PathVQA experiment completed successfully using the reduced subset. The validation run processed all 500 validation samples, and the trained adapter was backed up.

### Results Summary

| Dataset | Training Samples Used | Validation Samples Used | Training Loss | Validation Loss |
|---|---:|---:|---:|---:|
| VQA-RAD | 1,791 | 216 | 0.9294 | 0.8054 |
| SLAKE | 4,918 | 1,053 | 0.403944 | 0.192734 |
| PathVQA | 3,916 | 500 | 0.89736 | 0.66694 |

The results provide an initial comparison of the fine-tuning behavior across the three datasets. Direct comparison of losses should be interpreted cautiously because the datasets and their question-answer distributions differ.

---

## 8. LoRA Adapter Backups

The trained LoRA adapters were exported and downloaded for all three experiments.

| Dataset | Adapter Backup |
|---|---|
| VQA-RAD | `VQA_RAD_LoRA_Backup.zip` |
| SLAKE | `SLAKE_LoRA_Backup.zip` |
| PathVQA | `PathVQA_LoRA_Backup.zip` |

These adapter backups contain the fine-tuned LoRA weights and associated adapter configuration.

They are intended to preserve the results of the individual dataset experiments without requiring the complete base model weights to be stored in the repository.

The base model must still be available to load and use the adapters.

---

## 9. Project Structure

The repository is organized into separate components for preprocessing, dataset handling, model configuration, fine-tuning, evaluation, analysis, and testing.

```text
Medical-VQA-PEFT-Research/
│
├── analysis/
│   ├── compare_methods.py
│   ├── generate_graphs.py
│   └── statistical_analysis.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── preprocessing/
│   ├── analyze_duplicates.py
│   ├── check_split_leakage.py
│   ├── inspect_datasets.py
│   ├── prepare_data.py
│   ├── validate_data.py
│   └── validate_processed_data.py
│
├── tests/
│   ├── test_data.py
│   ├── test_evaluation.py
│   └── test_training.py
│
├── colab_train.py
├── dataset.py
├── dataset_loader.py
├── evaluate_pipeline.py
├── model_loader.py
├── peft_methods.py
├── robustness_pipeline.py
├── run_all.py
├── train_pipeline.py
├── utils.py
├── validate_llava_data.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The structure above highlights the main project components. Additional generated data, reports, checkpoints, and experiment files may be present locally or in the training environment.

---

## 10. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/AmitojOberoi/Medical-VQA-PEFT-Research.git

cd Medical-VQA-PEFT-Research
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU-based model fine-tuning, configure a compatible CUDA-enabled PyTorch environment and the required model dependencies.

**Note:** The complete LLaVA-Med 7B model requires substantial GPU memory and storage. A CPU-only environment is suitable for certain preprocessing and testing tasks but is not the intended environment for full model fine-tuning.

---

## 11. Running the Preprocessing Pipeline

The preprocessing scripts are located in the `preprocessing/` directory.

Examples:

```bash
python -m preprocessing.inspect_datasets
```

```bash
python -m preprocessing.prepare_data
```

```bash
python -m preprocessing.validate_data
```

```bash
python -m preprocessing.validate_processed_data
```

Additional scripts are provided for duplicate analysis and split leakage checking.

---

## 12. Testing

Automated tests were implemented to check important components of the project.

The test suite covers:

- Dataset loading and handling.
- Dataset validation.
- Data batching and collation.
- Evaluation metric calculations.
- Evaluation result saving.
- LoRA configuration.
- LoRA application.
- Trainable parameter summaries.

### Running Tests

Install pytest:

```bash
pip install pytest
```

Run the complete test suite:

```bash
python -m pytest -v
```

### Test Results

The completed test run reported:

| Test Module | Passed |
|---|---:|
| `test_data.py` | 23 |
| `test_evaluation.py` | 9 |
| `test_training.py` | 10 |
| **Total** | **42** |

**Final result: 42 passed in 17.91 seconds.**

These tests verify software components and expected behavior. They do not independently establish the clinical accuracy or diagnostic reliability of the trained model.

---

## 13. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained model integration |
| Hugging Face PEFT | Parameter-efficient fine-tuning |
| LLaVA-Med | Medical vision-language model |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Scikit-learn | Supporting evaluation and analysis |
| Kaggle Notebooks | GPU-based experiments |
| Git and GitHub | Version control and project hosting |

---

## 14. Limitations

The project has several limitations that should be considered when interpreting its results.

1. **Computational constraints:** Fine-tuning a 7B-parameter vision-language model requires substantial GPU resources.
2. **Reduced PathVQA experiment:** PathVQA training used 3,916 training samples and 500 validation samples rather than the complete processed splits.
3. **Limited training duration:** The recorded experiments used one training epoch.
4. **Evaluation scope:** The reported results include training and validation losses. They should not be interpreted as answer accuracy.
5. **Held-out test evaluation:** Test-set answer accuracy and other final performance metrics have not been established in the reported experiments.
6. **Medical reliability:** The model is a research prototype and is not intended for clinical diagnosis or medical decision-making.

---

## 15. Future Work

Potential extensions of this research include:

- Running additional training epochs.
- Evaluating the model on held-out test sets.
- Implementing answer-generation evaluation metrics.
- Comparing exact-match accuracy and other appropriate VQA metrics.
- Investigating the effect of different LoRA configurations.
- Conducting more extensive experiments on the complete PathVQA training split.
- Performing robustness and error analysis.
- Exploring cross-dataset generalization.
- Investigating model performance across different categories of medical questions.

---

## 16. Conclusion

This project explores parameter-efficient fine-tuning of a medical vision-language model for Medical Visual Question Answering.

Three datasets—VQA-RAD, SLAKE, and PathVQA—were processed and validated. A LLaVA-compatible data preparation pipeline was implemented, and LoRA-based fine-tuning experiments were completed using Kaggle GPU environments.

The experiments produced trained LoRA adapters for all three datasets. The software implementation was also tested through an automated test suite, with all 42 tests passing.

The work establishes a foundation for further research into parameter-efficient adaptation of large vision-language models for medical image question answering.

---

## 17. Repository

**GitHub:** https://github.com/AmitojOberoi/Medical-VQA-PEFT-Research

---

**Disclaimer:** This project is intended exclusively for academic and research purposes. It is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical judgment.