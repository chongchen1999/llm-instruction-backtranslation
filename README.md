# LLM Instruction Backtranslation Assignment Report

This repository contains the implementation details and results of a fine-tuning experiment using Low-Rank Adaptation (LoRA) to train a compact language model, **tinyllama-1.1b**, for instruction backtranslation tasks. The project aims to reverse-engineer instructions from responses, with a focus on optimizing resource usage.

## Table of Contents
- [Overview](#overview)
- [Setup and Requirements](#setup-and-requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
- [Instruction Generation](#instruction-generation)
- [Results and Analysis](#results-and-analysis)
- [Future Work](#future-work)
- [References](#references)

## Overview
This project demonstrates:
- Fine-tuning the **tinyllama-1.1b** model for instruction backtranslation.
- Generating high-quality instruction-response pairs using the fine-tuned model.

## Setup and Requirements

### Environment
- OS: Ubuntu
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- Python 3.8 or higher

### Dependencies
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
Dependencies include:
- `transformers`
- `datasets`
- `accelerate`
- `torch`
- `scipy`

### Repository Structure
```
.
├── data/
│   └── seed/                   # Preprocesses seed data
│   └── lima/                   # Generated instruction-response pairs
├── dataset/
│   ├── seed/                   # Seed data for training (JSONL format)
│   └── lima/                   # LIMA dataset (JSONL format)
├── models/
│   ├── tinyllama/              # Base model
│   └── tinyllama-backtranslation-model-Myx/  # Fine-tuned model
├── src/
│   ├── backtranslation_model_test.py    # Backtranslation model test
│   ├── base_model_test.py
│   ├── filter.py                # Filtering script
│   ├── train_Myx.py                # Training pipeline
│   ├── instruction_generator.py    # Instruction generation
│   └── evaluate.py             # Evaluation scripts
├── README.md                   # Project description
└── requirements.txt            # Dependency list
```

## Dataset Preparation

1. **Seed Dataset**: The **OASST1** dataset was used for training. Download it from [here](https://huggingface.co/datasets/OpenAssistant/oasst1).
2. **Preprocessing**: Convert dataset entries into `(response, instruction)` pairs and save in JSONL format.
   ```bash
   sh data/seed/download.sh
   python data/seed/convert.py
   ```

## Training Pipeline

1. Load the base model and preprocess the data:
   ```bash
   python src/train_Myx.py
   ```

2. Training parameters:
   - LoRA rank: 16
   - Batch size per GPU: 4 (gradient accumulation for effective batch size of 16)
   - Learning rate: 5e-4
   - Precision: FP16

3. Training outputs:
   - Fine-tuned weights: `models/tinyllama-backtranslation-model-Myx/`
   - Logs and checkpoints saved after each epoch.

## Instruction Generation

Generate instructions for filtered responses from the **LIMA** dataset:

1. Use the fine-tuned model to generate instructions:
   ```bash
   python src/instruction_generator.py
   ```
2. Filter dataset for single-turn conversations:
   ```bash
   python src/filter.py
   ```

## Results and Analysis

Sample generated instruction-response pairs:
| **Response**                                                                                 | **Generated Instruction**                            |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------|
| The question is relatively broad and one should take into account...                         | Is the brain cells migrating or not?                |
| There is a general historical trend. In the olden days...                                     | What is the historical trend in CPUs and processors? |
| Tor clients do not, in general, directly do DNS requests...                                   | How does the Tor client do the DNS requests?        |

For detailed results, see `data/lima/lima_instructions.json`.

## Future Work

- Extend to multi-turn instruction-response pairs for more complex scenarios.
- Train with larger datasets and models to improve performance.
- Experiment with advanced fine-tuning techniques like contrastive learning.

## References

- **Dataset**: [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **Base Model**: [TinyLLaMA-1.1b](https://huggingface.co/Tourist99/tinyllama)
- **Paper**: ["Self Alignment with Instruction Backtranslation"](https://arxiv.org/pdf/2308.06259.pdf)

## Huggingface Link
- **Base Model**: https://huggingface.co/Tourist99/tinyllama
- **Fine-tuned Model**: https://huggingface.co/Tourist99/tinyllama-backtranslation-model-Myx