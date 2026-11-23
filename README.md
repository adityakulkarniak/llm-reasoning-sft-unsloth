Markdown# llm-reasoning-sft-unsloth

End-to-end QLoRA Supervised Fine-Tuning (SFT) of `Llama-3.2-3B-Instruct` on `ServiceNow-AI/R1-Distill-SFT` (`v0`) using Unsloth on a Tesla T4 GPU, with GGUF (`Q8_0`) quantization and local Ollama deployment.

---

## Project Overview

Compact instruction-tuned language models often jump straight to direct answers without explicit intermediate verification steps. This project fine-tunes `unsloth/Llama-3.2-3B-Instruct` (`3,237,063,680` total parameters) on reasoning traces from `ServiceNow-AI/R1-Distill-SFT` (`v0`) to encourage structured, step-by-step reasoning (`<think>...</think>`) within the memory budget of a single NVIDIA Tesla T4 GPU (`14.563 GB` VRAM).

By freezing quantized base model weights and training only `24,313,856` adapter parameters (`0.75%` of total parameters) via `TRL.SFTTrainer` and Unsloth, the pipeline adapts reasoning behavior efficiently. Over `60` optimizer steps (effective batch size `8`, completed in `14m 35s` on a Tesla T4), training loss drops from `0.9101` at Step 1 to `0.4882` at Step 60, reaching a minimum logged loss of `0.3971` at Step 57. Afterward, the model saves LoRA adapters to `chintan-001-3B`, exports a merged `Q8_0` GGUF binary and Ollama `Modelfile` to `chintan-001-3B-GGUF_gguf`, and configures local serving via Ollama at `http://127.0.0.1:11434`.

---

## System Architecture & Pipeline

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Quantized Base Model & LoRA Injection                                 │
│    Backbone: unsloth/Llama-3.2-3B-Instruct (4-bit, max_seq_length=2048) │
│    LoRA: r=16, lora_alpha=16, lora_dropout=0, bias="none"              │
│    Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,          │
│    down_proj across 28 LlamaDecoderLayer blocks                          │
│    Trainable Params: 24,313,856 / 3,237,063,680 (0.75%)                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. Dataset Loading & Prompt Formatting                                   │
│    Dataset: ServiceNow-AI/R1-Distill-SFT (config="v0", split="train")  │
│    Scale: 171,647 rows across 3 Parquet shards (~555 MB)               │
│    Template: Reflective system prompt + <problem> + thought + solution  │
│    Stop Token: Appended tokenizer.eos_token to each formatted sample    │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. Supervised Fine-Tuning (TRL SFTTrainer + Unsloth)                     │
│    Batching: per_device_train_batch_size=2, grad_accum=4 (eff. = 8)   │
│    Schedule: max_steps=60, lr=2e-4 (linear, 5 warmup), adamw_8bit       │
│    Runtime: 1x NVIDIA Tesla T4 (14.563 GB VRAM, FP16, 14m 35s)         │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. Inference Evaluation & GGUF / Ollama Export                           │
│    Inference: FastLanguageModel.for_inference() + llama-3.1 template    │
│    Adapters: chintan-001-3B/                                            │
│    GGUF & Modelfile: Llama-3.2-3B-Instruct.Q8_0.gguf                    │
│    Local Server: Ollama runtime on [http://127.0.0.1:11434](http://127.0.0.1:11434)               │
└──────────────────────────────────────────────────────────────────────────┘
Dataset Description (ServiceNow-AI/R1-Distill-SFT)This pipeline uses the v0 configuration (train split) of ServiceNow-AI/R1-Distill-SFT, a dataset containing reasoning traces distilled from DeepSeek-R1 across math and problem-solving prompts (171,647 training examples across three Parquet shards).   ColumnTypeDescriptionidstringUnique sample identifier (id_0, id_1, ...).   problemstringInput math or logical problem statement placed inside <problem>...</problem>.   reannotated_assistant_contentstringStep-by-step reasoning trace (<think>...</think>) followed by a structured walkthrough.   solutionstringStep-by-step derivation with the final answer formatted in \boxed{...}.   sourcestringSource subset label such as orca_math or synthetic_math.   verified / quality_metricsnull / metadataOptional evaluation metadata fields present in the dataset schema.   Formatting Functionformatting_prompts_func maps each example into a "text" field using dataset.map(..., batched=True) and appends tokenizer.eos_token so the model learns clean stopping boundaries.   PlaintextYou are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{problem}
</problem>

{reannotated_assistant_content}
{solution}<EOS_TOKEN>
Fine-Tuning Methodology (QLoRA & Unsloth)Four-bit quantization (load_in_4bit=True) reduces memory demand so a 3.24B parameter model fits alongside gradients, activations, and optimizer states on a 14.563 GB NVIDIA Tesla T4 GPU (dtype=None selects FP16 with Bfloat16=False on T4).   Low-Rank Adaptation (LoRA) keeps base model weights $W_0$ frozen and trains low-rank decomposition matrices $A$ and $B$:   $$W' = W_0 + \frac{\alpha}{r}(A \cdot B)$$Using r=16, lora_alpha=16 ($\alpha/r = 1.0$), lora_dropout=0, bias="none", and use_gradient_checkpointing="unsloth", Unsloth patches q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj across all 28 LlamaDecoderLayer blocks while updating only 24,313,856 parameters (0.75%).   Supervised fine-tuning runs through TRL.SFTTrainer with per_device_train_batch_size=2 and gradient_accumulation_steps=4 for an effective batch size of 8, using optim="adamw_8bit", learning_rate=2e-4, warmup_steps=5, lr_scheduler_type="linear", and weight_decay=0.01 across max_steps=60 (480 sequences).   Training Configuration & ResultsThe run executed on 1x NVIDIA Tesla T4 (14.563 GB VRAM, CUDA Capability 7.5) with Unsloth 2026.11.7, PyTorch 2.14.1+cu130, Transformers 5.17.0, and TRL SFTTrainer, finishing 60 steps in 14m 35s.   StepTraining LossNotesStep 10.910080Initial training step   Step 100.686514Early drop as reasoning structure is learned   Step 210.464000Dropped below 0.50   Step 450.434314Low loss point later in the run   Step 570.397131Lowest recorded loss during the 60-step run   Step 600.488191Final logged step (outputs/checkpoint-60 saved)   Inference Behavior & Error AnalysisPost-training inference ran via FastLanguageModel.for_inference(model) using the llama-3.1 chat template (max_new_tokens=1024, temperature=1.5, min_p=0.1) on the prompt "How many 'r's are present in 'strawberry'?".   The model adopted a reflective, step-by-step monologue ("Alright, let me try to figure out...", "Let me break it down...", "Let me recount...")[cite: 1]. However, it counted 2 'r's instead of 3 and hit the 1024-token limit during repeated self-checks because subword tokenizers do not split words into single characters, max_steps=60 is a short format-adaptation run, and temperature=1.5 increases generation drift[cite: 1]. In production systems, exact character counting or arithmetic is better offloaded to tool calling or Python execution alongside RAG rather than relying on token generation alone.Model Serialization, GGUF Conversion & Ollama DeploymentPython# Save LoRA adapters & tokenizer
model.save_pretrained("chintan-001-3B")
tokenizer.save_pretrained("chintan-001-3B")

# Merge to 16-bit & export GGUF (Q8_0) + Ollama Modelfile
model.save_pretrained_gguf("chintan-001-3B-GGUF", tokenizer)
Bash# Serve and run locally via Ollama
ollama serve &
ollama create chintan-001-3b -f chintan-001-3B-GGUF_gguf/Modelfile
ollama run chintan-001-3b
Repository Structure & SetupPlaintext.
├── finetuningipynb.ipynb          # Colab notebook: QLoRA SFT, inference, GGUF export & Ollama setup
├── README.md                      # Project documentation
├── chintan-001-3B/                # Saved LoRA adapter weights & tokenizer files (if tracked)
└── chintan-001-3B-GGUF_gguf/      # Exported Q8_0 GGUF artifact & Ollama Modelfile (if tracked)
Install Unsloth and open finetuningipynb.ipynb in Google Colab with a Tesla T4 GPU runtime (Runtime -> Change runtime type -> T4 GPU) to run the cells sequentially[cite: 1].Bashpip install -q unsloth
