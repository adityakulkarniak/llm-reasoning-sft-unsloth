# llm-reasoning-sft-unsloth
End-to-end QLoRA Supervised Fine-Tuning (SFT) of Llama-3.2-3B-Instruct on ServiceNow-AI/R1-Distill-SFT using Unsloth on a T4 GPU, with GGUF Q8_0 quantization and local Ollama deployment.

# Llama-3.2-3B Reasoning Fine-Tuning (QLoRA + Unsloth + GGUF/Ollama)

An end-to-end Large Language Model (LLM) post-training pipeline that adapts **`unsloth/Llama-3.2-3B-Instruct`** for structured, step-by-step chain-of-thought reasoning (`<think>...</think>`) using **4-bit QLoRA Supervised Fine-Tuning (SFT)** on a single **NVIDIA Tesla T4 GPU (14.56 GB VRAM)**, followed by **GGUF (`Q8_0`) export** and local serving via **Ollama**.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture & Pipeline](#2-system-architecture--pipeline)
3. [Dataset Description (`ServiceNow-AI/R1-Distill-SFT`)](#3-dataset-description-servicenow-air1-distill-sft)
4. [Fine-Tuning Methodology (QLoRA & Unsloth)](#4-fine-tuning-methodology-qlora--unsloth)
5. [Training Configuration & Results](#5-training-configuration--results)
6. [Inference Behavior & Error Analysis](#6-inference-behavior--error-analysis)
7. [Model Serialization, GGUF Conversion & Ollama Deployment](#7-model-serialization-gguf-conversion--ollama-deployment)
8. [Repository Structure & Setup](#8-repository-structure--setup)
9. [Author & Attribution](#9-author--attribution)

---

## 1. Project Overview

Standard instruction-tuned compact models (such as `Llama-3.2-3B-Instruct`) typically jump straight to direct answers without explicit intermediate verification steps. This project demonstrates how to instill **iterative, reflective chain-of-thought reasoning** into a 3-billion-parameter model under tight consumer/cloud GPU memory constraints.

Key engineering highlights:
* **Memory-Efficient Post-Training:** Loaded the 3.24B-parameter base model in 4-bit precision (`bitsandbytes` NF4) and trained only **0.75% of total parameters** (`24,313,856` trainable parameters) via Low-Rank Adaptation (LoRA).
* **Reasoning Distillation:** Formatted reasoning traces (`problem` $\rightarrow$ `<think>` trace $\rightarrow$ `\boxed{}` solution) with explicit `EOS` termination tokens.
* **Production Edge Packaging:** Merged trained LoRA adapters back into 16-bit weights, converted the model into a single **`Q8_0` GGUF** binary via `llama.cpp`, and configured an **Ollama `Modelfile`** for local/offline inference.

---

## 2. System Architecture & Pipeline

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Base Model Ingestion                                                  │
│    unsloth/Llama-3.2-3B-Instruct (4-bit quantized, max_seq_length=2048)  │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. PEFT / LoRA Injection                                                 │
│    Rank r=16, alpha=16 across Q, K, V, O + Gate, Up, Down (28 layers)    │
│    Trainable: 24.31M / 3.24B parameters (0.75%)                          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. Dataset Formatting (ServiceNow-AI/R1-Distill-SFT, config='v0')        │
│    171,647 reasoning examples mapped to reflective prompt + EOS_TOKEN    │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. Supervised Fine-Tuning (TRL SFTTrainer + Unsloth Kernels)             │
│    Effective Batch Size = 8 (2 per device × 4 grad accum), 60 steps      │
│    Loss reduced: 0.9101 (Step 1) ──► 0.4882 (Step 60) in 14m 35s on T4   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. Export & Local Serving                                                │
│    LoRA Adapters (chintan-001-3B) ──► Merged FP16 ──► GGUF (Q8_0)        │
│    Served locally via Ollama REST API (127.0.0.1:11434)                  │
└──────────────────────────────────────────────────────────────────────────┘
