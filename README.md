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

```text
You are a reflective assistant engaging in thorough,
iterative reasoning, mimicking human stream-of-consciousness thinking.
Your approach emphasizes exploration, self-doubt,
and continuous refinement before coming up with an answer.

<problem>
{problem}
</problem>

{reannotated_assistant_content}

{solution}
```

The tokenizer's EOS token is appended to every formatted sample to establish sequence termination boundaries during training.

**Formatting implementation:**

```python
def formatting_prompts_func(examples):
    texts = []

    for problem, reasoning, solution in zip(
        examples["problem"],
        examples["reannotated_assistant_content"],
        examples["solution"]
    ):
        text = f"""You are a reflective assistant engaging in thorough,
iterative reasoning, mimicking human stream-of-consciousness thinking.

<problem>
{problem}
</problem>

{reasoning}

{solution}"""

        texts.append(text + tokenizer.eos_token)

    return {"text": texts}


dataset = dataset.map(
    formatting_prompts_func,
    batched=True
)
```

---

## Fine-Tuning Methodology

### 1. Quantized Model Loading

The base model is loaded using Unsloth's `FastLanguageModel`.

Four-bit quantization reduces the memory footprint of pretrained model weights, allowing fine-tuning within the available GPU memory budget.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
```

### 2. LoRA Adapter Configuration

Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning by freezing the pretrained weights and learning additional low-rank matrices.

Instead of directly updating the original weight matrix, LoRA represents the adapted weights as:

\[
W' = W_0 + \frac{\alpha}{r}(BA)
\]

Where:

- \(W_0\): Frozen pretrained weight matrix.
- \(A\) and \(B\): Trainable low-rank matrices.
- \(r\): LoRA rank.
- \(\alpha\): LoRA scaling parameter.

The implementation uses the following configuration:

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Bias | none |
| Gradient Checkpointing | unsloth |
| Random State | 3407 |

**LoRA target modules:**

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

**Implementation:**

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### 3. Trainable Parameter Efficiency

The fine-tuning process updates only the LoRA adapter parameters.

| Parameter Metric | Count |
|---|---:|
| Total Model Parameters | 3,237,063,680 |
| Trainable Parameters | 24,313,856 |
| Frozen Parameters | 3,212,749,824 |
| Trainable Percentage | 0.75% |

This significantly reduces the number of parameters requiring gradient updates compared with full-parameter fine-tuning.

### 4. Supervised Fine-Tuning

The model is trained using Hugging Face TRL's `SFTTrainer`, with Unsloth optimizations.

**Training configuration:**

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    fp16=True,
    bf16=False,
    report_to="none",
    max_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer_stats = trainer.train()
```

**Batch size calculation:**

\[
\text{Effective Batch Size}
=
2 \times 4 \times 1 = 8
\]

Where:

- Per-device batch size = 2
- Gradient accumulation steps = 4
- Number of GPUs = 1

Therefore, the effective batch size is 8 sequences per optimizer update.

---

## Training Configuration

### Hardware Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 14.563 GB |
| CUDA Capability | 7.5 |
| Precision | FP16 |
| GPU Count | 1 |

### Software Environment

| Component | Version |
|---|---|
| Unsloth | 2026.11.7 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.17.0 |
| Training Framework | TRL SFTTrainer |

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Maximum Sequence Length | 2048 |
| Training Steps | 60 |
| Learning Rate | 0.0002 |
| Optimizer | AdamW 8-bit |
| Scheduler | Linear |
| Warmup Steps | 5 |
| Weight Decay | 0.01 |
| Per-Device Batch Size | 2 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 8 |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Quantization | 4-bit |
| Precision | FP16 |

---

## Training Results

The fine-tuning run completed 60 optimizer steps in approximately **14 minutes and 35 seconds** on a single NVIDIA Tesla T4 GPU.

### Training Loss Progression

| Step | Training Loss | Observation |
|---:|---:|---|
| 1 | 0.9101 | Initial training loss |
| 10 | 0.6865 | Early loss reduction |
| 21 | 0.4640 | Loss dropped below 0.50 |
| 45 | 0.4343 | Continued optimization |
| 57 | 0.3971 | Minimum logged loss |
| 60 | 0.4882 | Final logged training loss |

### Training Loss Visualization

```text
Training Loss

Step  1  | 0.9101 | ██████████████████
Step 10  | 0.6865 | ██████████████
Step 21  | 0.4640 | █████████
Step 45  | 0.4343 | █████████
Step 57  | 0.3971 | ████████
Step 60  | 0.4882 | ██████████
```

The logged training loss decreased substantially during the run.

The minimum logged loss was 0.3971 at step 57, while the final logged loss was 0.4882 at step 60.

**Important:** These are training-loss observations, not evidence of improved reasoning accuracy or generalization. A separate held-out evaluation is required to establish actual task performance.

---

## Inference and Evaluation

After training, the model is switched into inference mode using Unsloth.

```python
FastLanguageModel.for_inference(model)
```

### Inference Configuration

| Parameter | Value |
|---|---|
| Maximum New Tokens | 1024 |
| Temperature | 1.5 |
| Minimum Probability (`min_p`) | 0.1 |
| Chat Template | llama-3.1 |

### Evaluation Prompt

```text
How many 'r's are present in 'strawberry'?
```

### Observed Behavior

The fine-tuned model generated a reflective, step-by-step response, using phrases such as:

- "Alright, let me try to figure out..."
- "Let me break it down..."
- "Let me recount..."

However, the model incorrectly counted **2 occurrences of 'r' instead of 3** and continued generating repeated self-checking text until reaching the 1024-token generation limit.

### Error Analysis

This experiment highlights several important limitations:

1. **Character-Level Precision:** Language models operate over token sequences rather than inherently performing exact character-by-character computation.

2. **Limited Training Steps:** Sixty optimizer steps are insufficient to establish robust reasoning improvements or generalization.

3. **Generation Instability:** A temperature of 1.5 can increase output variability and contribute to repetitive or inconsistent responses.

4. **Reasoning vs. Correctness:** Producing longer reasoning traces does not guarantee that the final answer is correct.

5. **Evaluation Requirements:** Training loss alone cannot establish reasoning capability. Held-out benchmarks and task-specific accuracy measurements are necessary.

### Engineering Observation

For exact character counting, arithmetic, and deterministic computations, a production LLM application should use external tools such as Python execution or structured tool calling rather than relying exclusively on generated text.

This experiment demonstrates the distinction between learning a reasoning-oriented response format and developing reliable problem-solving capability.

---

## Model Export and Ollama Deployment

After fine-tuning, the trained LoRA adapters and tokenizer are saved locally.

### 1. Save LoRA Adapters

```python
model.save_pretrained("chintan-001-3B")
tokenizer.save_pretrained("chintan-001-3B")
```

This preserves the fine-tuned adapter weights separately from the original pretrained model.

### 2. GGUF Export

The fine-tuned model is exported into GGUF format using Unsloth.

```python
model.save_pretrained_gguf(
    "chintan-001-3B-GGUF",
    tokenizer,
)
```

The export configuration targets `Q8_0` quantization.

GGUF provides a model format suitable for local inference runtimes such as llama.cpp and Ollama.

### 3. Ollama Deployment

Start the Ollama service:

```bash
ollama serve
```

Create a local model from the generated Modelfile:

```bash
ollama create chintan-001-3b \
  -f chintan-001-3B-GGUF_gguf/Modelfile
```

Run the model:

```bash
ollama run chintan-001-3b
```

The model can then be accessed through the local Ollama runtime.

**Default local API endpoint:**

```text
http://127.0.0.1:11434
```

### 4. Test Local Inference

Example API request:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "chintan-001-3b",
    "prompt": "Explain the difference between supervised and unsupervised learning.",
    "stream": false
  }'
```

This sends a generation request to the locally running model.

---

## Project Structure

```text
llm-reasoning-sft-unsloth/
│
├── finetuningipynb.ipynb
│   └── End-to-end Google Colab notebook
│       ├── Environment setup
│       ├── Dataset loading
│       ├── Prompt formatting
│       ├── 4-bit model loading
│       ├── LoRA configuration
│       ├── SFT training
│       ├── Training loss monitoring
│       ├── Inference and evaluation
│       ├── Adapter serialization
│       └── GGUF export
│
├── README.md
│
├── chintan-001-3B/
│   └── Saved LoRA adapters and tokenizer files
│
└── chintan-001-3B-GGUF_gguf/
    ├── Llama-3.2-3B-Instruct.Q8_0.gguf
    └── Modelfile
```

*Note: Model artifacts are generated during execution. Large model files are not necessarily tracked in Git.*

---

## Installation and Execution

### Prerequisites

- Python
- Google Colab
- NVIDIA Tesla T4 GPU or compatible CUDA GPU
- Hugging Face account and access to the selected dataset
- Ollama installed locally for deployment

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-reasoning-sft-unsloth.git

cd llm-reasoning-sft-unsloth
```

### Step 2: Open Google Colab

Open `finetuningipynb.ipynb` in Google Colab.

Navigate to:

```text
Runtime → Change runtime type → T4 GPU
```

### Step 3: Install Unsloth

Run the installation cell in the notebook:

```python
!pip install -q unsloth
```

### Step 4: Execute the Notebook

Execute notebook cells sequentially to:

1. Install and configure dependencies.
2. Load the Hugging Face dataset.
3. Format the training prompts.
4. Load the quantized pretrained model.
5. Configure LoRA adapters.
6. Execute supervised fine-tuning.
7. Inspect training metrics.
8. Run inference tests.
9. Save the fine-tuned adapters.
10. Export the GGUF model.

### Step 5: Deploy Using Ollama

Download the exported GGUF artifact to your local machine and follow the Ollama deployment instructions above.

---

## Limitations and Engineering Observations

This project is an experimental SFT implementation rather than a production-ready reasoning model.

Key limitations include:

- **Short Training Duration:** Only 60 optimizer steps were performed. This is a limited format-adaptation experiment, not comprehensive model training.
- **No Held-Out Benchmark:** No independent test-set accuracy or reasoning benchmark is reported.
- **Training Loss Limitations:** A decreasing training loss does not necessarily translate into improved reasoning accuracy.
- **Generation Repetition:** The inference experiment exhibited repetitive self-checking and token-limit exhaustion.
- **High Temperature:** Temperature 1.5 is relatively aggressive for deterministic mathematical evaluation.
- **Tool Dependence:** Exact calculations and character-level operations should be handled using deterministic external tools when correctness matters.
- **Deployment Validation:** GGUF export and Ollama configuration do not, by themselves, establish production readiness.

---

## Future Improvements

- Increase training steps and evaluate convergence.
- Introduce a dedicated train-validation-test split.
- Benchmark the base model against the fine-tuned model.
- Measure exact-match accuracy on mathematical and logical reasoning tasks.
- Evaluate output quality, repetition, and response length.
- Experiment with lower inference temperatures.
- Implement structured tool calling for arithmetic and deterministic tasks.
- Compare Q4_K_M, Q5_K_M, and Q8_0 GGUF quantization.
- Build a FastAPI inference service around the locally deployed model.
- Integrate the fine-tuned model into a retrieval-augmented generation (RAG) pipeline.

---

## Key Takeaways

This project demonstrates an end-to-end parameter-efficient fine-tuning workflow for a compact instruction-tuned language model under constrained GPU resources.

The implementation covers:

- Quantized model loading using QLoRA.
- Low-Rank Adaptation for parameter-efficient training.
- Reasoning-oriented dataset preparation.
- Supervised fine-tuning with Unsloth and TRL.
- Training-loss monitoring and inference error analysis.
- GGUF model serialization.
- Local deployment using Ollama.

**The central engineering takeaway:** Teaching a language model to produce structured reasoning traces is not the same as making it reliably solve reasoning problems. Robust evaluation, deterministic tool integration, and controlled inference are essential for building dependable LLM applications.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face](https://huggingface.co/)
- [ServiceNow-AI/R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
- [Meta Llama](https://huggingface.co/meta-llama)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [Ollama](https://ollama.com/)

---


**Project:** LLM Reasoning Fine-Tuning with QLoRA, Unsloth & Ollama
