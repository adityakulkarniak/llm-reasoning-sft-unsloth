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

Dataset Description: ServiceNow-AI/R1-Distill-SFT (v0)This notebook loads the v0 (train split) of ServiceNow-AI/R1-Distill-SFT, a reasoning dataset distilled from DeepSeek-R1 reasoning traces across math and problem-solving prompts (171,647 training rows across three Parquet files).   ColumnTypeDescriptionidstringUnique sample identifier (id_0, id_1, ...).   problemstringInput math or logical problem statement placed inside <problem>...</problem>.   reannotated_assistant_contentstringStep-by-step reasoning trace (<think>...</think>) followed by a structured walkthrough.   solutionstringStep-by-step solution with the final result in \boxed{...} notation.   sourcestringSource subset label (e.g., orca_math, synthetic_math).   verified / quality_metricsnull / metadataOptional evaluation metadata fields present in the dataset.   Formatting Functionformatting_prompts_func formats each example into "text" with dataset.map(..., batched=True) and appends tokenizer.eos_token:   PlaintextYou are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{problem}
</problem>

{reannotated_assistant_content}
{solution}<EOS_TOKEN>
Core Concepts & Implementation1. 4-Bit QuantizationLoading a ~3.24B parameter model in 32-bit floats (FP32) requires ~12 GB for weights alone, leaving limited room on a 14.563 GB Tesla T4 GPU once activations, gradients, and optimizer states are allocated. FastLanguageModel.from_pretrained(..., load_in_4bit=True, dtype=None) loads the 4-bit model (unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit) and uses FP16 on T4 (Bfloat16 = False).   2. Low-Rank Adaptation (LoRA)Instead of updating all 3.24B parameters, LoRA freezes base weights $W_0$ and trains low-rank matrices $A$ and $B$:   $$W' = W_0 + \frac{\alpha}{r}(A \cdot B)$$Applied settings: r = 16, lora_alpha = 16 ($\alpha / r = 1.0$), lora_dropout = 0, bias = "none", and use_gradient_checkpointing = "unsloth" across q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj on all 28 LlamaDecoderLayer blocks.   Trainable footprint: 24,313,856 parameters (0.75%) updated during training.   3. Supervised Fine-Tuning (TRL.SFTTrainer)Effective batch size: per_device_train_batch_size = 2 with gradient_accumulation_steps = 4 on 1 GPU gives an effective batch size of $2 \times 4 \times 1 = 8$.   Optimizer & schedule: optim = "adamw_8bit", learning_rate = 2e-4, warmup_steps = 5, lr_scheduler_type = "linear", and weight_decay = 0.01 over max_steps = 60 (480 sequences).   Training Results & Inference AnalysisLoss Progression (60 Steps · 14m 35s)StepTraining LossNotesStep 10.910080Initial training step   Step 100.686514Early drop as reasoning format is learned   Step 210.464000Dropped below 0.50   Step 450.434314Low loss point later in the run   Step 570.397131Lowest recorded loss across the 60 steps   Step 600.488191Final logged step (outputs/checkpoint-60 saved)[cite: 1]Inference Test & FindingsUsing FastLanguageModel.for_inference(model) with the llama-3.1 chat template (max_new_tokens = 1024, temperature = 1.5, min_p = 0.1), the model was tested on[cite: 1]:"How many 'r's are present in 'strawberry'?"Observed behavior: The model produced a reflective, step-by-step monologue ("Alright, let me try to figure out...", "Let me break it down...", "Let me recount...")[cite: 1].Failure mode & takeaway: It counted 2 'r's instead of 3 and reached the 1024-token cap during repeated self-checks[cite: 1]. Standard subword tokenizers do not split words into single characters, max_steps = 60 is a brief format-adaptation run, and temperature = 1.5 increases generation drift[cite: 1]. In production applications, exact character counting or math is better handled via tool calling / Python execution or RAG rather than relying on token generation alone.Export & Local Ollama UsagePython# Save LoRA adapters & tokenizer
model.save_pretrained("chintan-001-3B")
tokenizer.save_pretrained("chintan-001-3B")

# Merge to 16-bit & export GGUF (Q8_0) + Ollama Modelfile
model.save_pretrained_gguf("chintan-001-3B-GGUF", tokenizer)
Bash# Serve and run locally via Ollama
ollama serve &
ollama create chintan-001-3b -f chintan-001-3B-GGUF_gguf/Modelfile
ollama run chintan-001-3b
Repository StructurePlaintext.
├── finetuningipynb.ipynb          # Colab notebook: QLoRA SFT, inference, GGUF export & Ollama setup
├── README.md                      # Project documentation
├── chintan-001-3B/                # Saved LoRA adapter weights & tokenizer files (if tracked)
└── chintan-001-3B-GGUF_gguf/      # Exported Q8_0 GGUF artifact & Ollama Modelfile (if tracked)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. Export & Local Serving                                                │
│    LoRA Adapters (chintan-001-3B) ──► Merged FP16 ──► GGUF (Q8_0)        │
│    Served locally via Ollama REST API (127.0.0.1:11434)                  │
└──────────────────────────────────────────────────────────────────────────┘
