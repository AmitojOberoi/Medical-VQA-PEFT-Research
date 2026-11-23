# Medical VQA - Experimental Results

## Base Model
- Model: microsoft/llava-med-v1.5-mistral-7b
- Fine-tuning: LoRA
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: q_proj, v_proj
- Epochs: 1
- GPU: NVIDIA T4

## Experiment Results

| Dataset | Training Samples | Validation Samples | Training Loss | Validation Loss | Training Time | Peak GPU |
|---|---:|---:|---:|---:|---|---:|
| VQA-RAD | 1791 | 216 | 0.929398 | 0.805412 | 2h 09m 27s | 7.36 GB |
| SLAKE | 4918 | 1053 | 0.403944 | 0.192734 | 6h 37m 16s | 7.37 GB |
| PathVQA | 3916 | 500 | 0.897355 | 0.666940 | 4h 46m 52s | 7.38 GB |

## PathVQA Subset Note
The full processed PathVQA dataset contains 19,648 training and 6,259 validation records. Due to computational constraints, the experiment used a reduced subset of 3,916 training and 500 validation records.

## Adapter Backups
- VQA_RAD_backup.zip
- SLAKE_LoRA_Backup.zip
- PathVQA_LoRA_Backup.zip

## Interpretation
All three experiments completed and generated LoRA adapter backups. Training and validation losses are optimization metrics, not answer accuracy.

Held-out test accuracy and clinical reliability have not yet been established.

