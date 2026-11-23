# LLM Reasoning Fine-Tuning with QLoRA, Unsloth & Ollama

End-to-end Supervised Fine-Tuning (SFT) of **Llama-3.2-3B-Instruct** using QLoRA and Unsloth on a single NVIDIA Tesla T4 GPU. The project uses reasoning traces from `ServiceNow-AI/R1-Distill-SFT` (`v0`), trains lightweight LoRA adapters, exports the fine-tuned model to GGUF (`Q8_0`), and deploys it locally using Ollama.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset Description](#dataset-description)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Training Configuration](#training-configuration)
- [Training Results](#training-results)
- [Inference and Evaluation](#inference-and-evaluation)
- [Model Export and Ollama Deployment](#model-export-and-ollama-deployment)
- [Project Structure](#project-structure)
- [Installation and Execution](#installation-and-execution)
- [Limitations and Engineering Observations](#limitations-and-engineering-observations)
- [Future Improvements](#future-improvements)

---

## Project Overview

Compact instruction-tuned language models can produce direct answers without consistently following structured reasoning patterns. This project explores whether a smaller language model can be adapted to produce more structured, step-by-step responses through supervised fine-tuning on distilled reasoning data.

The base model, `unsloth/Llama-3.2-3B-Instruct`, contains approximately **3.24 billion parameters**. It is fine-tuned on the `ServiceNow-AI/R1-Distill-SFT` dataset using **4-bit Quantized Low-Rank Adaptation (QLoRA)**.

Instead of updating all model parameters, the training pipeline freezes the quantized base model and updates only lightweight LoRA adapter weights.

The complete workflow covers:

1. Loading a pretrained Llama-3.2-3B-Instruct model in 4-bit precision.
2. Preparing and formatting reasoning-oriented supervised training data.
3. Injecting LoRA adapters into attention and MLP projection layers.
4. Performing supervised fine-tuning using TRL's `SFTTrainer`.
5. Evaluating the fine-tuned model through local inference.
6. Saving the trained LoRA adapters.
7. Exporting the model to GGUF (`Q8_0`) format.
8. Deploying the exported model locally using Ollama.

### Key Project Metrics

| Metric | Value |
|---|---|
| Base Model | Llama-3.2-3B-Instruct |
| Total Parameters | 3,237,063,680 |
| Trainable Parameters | 24,313,856 |
| Trainable Parameter Ratio | 0.75% |
| Dataset | ServiceNow-AI/R1-Distill-SFT |
| Dataset Configuration | v0 |
| Training Examples Available | 171,647 |
| Fine-Tuning Method | QLoRA + LoRA |
| Quantization During Training | 4-bit |
| GPU | NVIDIA Tesla T4 |
| GPU VRAM | 14.563 GB |
| Training Steps | 60 |
| Effective Batch Size | 8 |
| Training Duration | 14 minutes 35 seconds |
| Initial Training Loss | 0.9101 |
| Final Training Loss | 0.4882 |
| Minimum Logged Training Loss | 0.3971 |
| Export Format | GGUF Q8_0 |
| Local Deployment | Ollama |

---

## Project Objectives

- Adapt a compact instruction-following model using reasoning-oriented supervised fine-tuning.
- Understand parameter-efficient fine-tuning using QLoRA and LoRA.
- Train a 3B parameter model within a single Tesla T4 GPU memory budget.
- Explore reasoning-trace formatting and EOS-based sequence termination.
- Monitor training loss and inspect inference behavior.
- Serialize trained adapters and export a deployable GGUF model.
- Run the fine-tuned model locally through Ollama.

---

## System Architecture

```text
                   PRETRAINED MODEL
                          |
                          v
              Llama-3.2-3B-Instruct
                          |
                          v
                4-Bit Quantization
                   (QLoRA Loading)
                          |
                          v
                LoRA Adapter Injection
                          |
             +------------+------------+
             |                         |
             v                         v
       Attention Layers            MLP Layers
       q_proj, k_proj              gate_proj
       v_proj, o_proj              up_proj
                                   down_proj
             |                         |
             +------------+------------+
                          |
                          v
                  DATASET PIPELINE
                          |
                          v
             ServiceNow-AI/R1-Distill-SFT
                          |
                          v
                Prompt Formatting
                          |
                          v
                 SUPERVISED SFT
                          |
                          v
                 TRL SFTTrainer
                   + Unsloth
                          |
                          v
                Trained LoRA Adapters
                          |
                          v
                Model Serialization
                          |
                          v
                 GGUF Export (Q8_0)
                          |
                          v
                   Ollama Runtime
                          |
                          v
                Local Model Inference
```

### Pipeline Stages

| Stage | Implementation |
|---|---|
| Base Model Loading | Unsloth FastLanguageModel |
| Quantization | 4-bit loading |
| Adapter Injection | LoRA |
| Dataset Preparation | Hugging Face Datasets |
| Prompt Formatting | Custom formatting function |
| Fine-Tuning | TRL SFTTrainer |
| Optimization | AdamW 8-bit |
| Memory Optimization | Gradient checkpointing |
| Inference | Unsloth FastLanguageModel |
| Model Export | GGUF Q8_0 |
| Deployment | Ollama |

---

## Technology Stack

| Category | Tools / Technologies |
|---|---|
| Programming Language | Python |
| Model Architecture | Llama-3.2-3B-Instruct |
| Fine-Tuning | QLoRA, LoRA |
| Fine-Tuning Framework | Unsloth |
| Training Framework | Hugging Face TRL |
| Model Hub | Hugging Face |
| Dataset Processing | Hugging Face Datasets |
| Deep Learning | PyTorch |
| Model Utilities | Transformers |
| GPU Acceleration | CUDA |
| Optimizer | AdamW 8-bit |
| Model Serialization | GGUF |
| Quantization | 4-bit, Q8_0 |
| Local Inference | Ollama |
| Development Environment | Google Colab |

---

## Dataset Description

**Dataset:** [`ServiceNow-AI/R1-Distill-SFT`](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)

The project uses the `v0` configuration and the `train` split.

The dataset contains reasoning-oriented examples distilled from DeepSeek-R1, covering mathematical and problem-solving tasks.

### Dataset Specifications

| Attribute | Description |
|---|---|
| Dataset Name | ServiceNow-AI/R1-Distill-SFT |
| Configuration | v0 |
| Split | train |
| Available Examples | 171,647 |
| Storage Format | Parquet |
| Number of Parquet Shards | 3 |
| Approximate Dataset Size | 555 MB |

### Dataset Columns

| Column | Type | Description |
|---|---|---|
| `id` | String | Unique sample identifier |
| `problem` | String | Mathematical or logical problem statement |
| `reannotated_assistant_content` | String | Reasoning trace and structured walkthrough |
| `solution` | String | Solution containing the final answer |
| `source` | String | Source subset identifier |
| `verified` | Metadata | Optional verification metadata |
| `quality_metrics` | Metadata | Optional quality-related metadata |

### Prompt Formatting

Each training example is transformed into a structured conversational-style text sequence.

The formatting function combines the problem statement, reasoning trace, and final solution.

**Training prompt template:**

