# Nexara

<!-- badges:start -->
[![CI](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/ci.yml/badge.svg)](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/ci.yml)
[![Docs](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/docs.yml/badge.svg)](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/docs.yml)
[![Nightly](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/nightly.yml/badge.svg)](https://github.com/NexaraAI/nexara-nano-100m-chat/actions/workflows/nightly.yml)
<!-- badges:end -->

Nexara is a small, experimental conversational language model project built from scratch using PyTorch and licensed under the MIT License. 

This repository is designed for educational and experimental purposes. It does **not** reuse pre-trained weights or commercial tokenizers; all models and vocabularies are trained directly from scratch.

> [!WARNING]
> This model is extremely small (~100M parameters) and trained on a tiny corpus. It is **not** guaranteed to perform well in production environments and is highly prone to factual hallucinations.

## Current Status

The codebase contains a fully functional training and fine-tuning pipeline:

- Custom BPE tokenizer training.
- Autoregressive JSONL token block loading and memory-mapped binary caching.
- Decoder-only transformer architecture with RoPE, causal self-attention, tied embeddings, and layer normalization.
- Training loops with validation, checkpointing, gradient accumulation, mixed-precision, and cosine decay learning rate schedules.
- SFT (Supervised Fine-Tuning) dataset balancing, templates, masking, and evaluation suite.
- Chat CLI interface for interactive local testing.

The primary experimental line is **Nexara Nano 100M-Chat** (~97.5M parameters, 12 layers, 12 attention heads, 768 embedding dimension, and 512 context length). Pretraining and Stage 2 SFT have been completed successfully.


## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data Layout

Raw TinyStories downloads go here:

```text
datasets/raw/TinyStories-train.txt
datasets/raw/TinyStories-valid.txt
```

Processed JSONL and token caches go under `datasets/processed/`. Generated data
artifacts are intentionally ignored by Git.

## Dataset Pipeline

Plan downloads without writing files:

```powershell
python -m scripts.download_tinystories --variant original --split all --dry-run
```

Download raw TinyStories text files:

```powershell
python -m scripts.download_tinystories --variant original --split all
```

Preprocess raw text into JSONL:

```powershell
python -m scripts.preprocess_tinystories --config configs/stage1_tinystories.toml
```

Generate dataset report JSON:

```powershell
python -m scripts.dataset_report --config configs/stage1_tinystories.toml
```

## Train Tokenizer

```powershell
python -m scripts.train_tokenizer --config configs/stage1_tinystories.toml --build-cache --overwrite
```

This writes uint32 token caches and outputs tokenizer reports, metadata, a vocabulary report, and sample encodings.

Generate tokenizer frequency analysis:

```powershell
python -m scripts.tokenizer_report --config configs/stage1_tinystories.toml
```

## Runtime Smoke Checks

```powershell
python -m scripts.verify_parameter_count --config configs/stage1_tinystories.toml
python -m scripts.tokenizer_smoke --output-dir logs/tokenizer_smoke
python -m scripts.benchmark_forward --tiny --iterations 5 --output logs/benchmark/forward.json
python -m scripts.train_smoke --output-dir logs/train_smoke
python -m scripts.run_overfit_validation --output-dir logs/overfit --steps 300 --subset-size 100 --experiment-name phase1_3_overfit_validation
```

## Post-Hoc Analysis

Plot loss and gradient curves from a completed run:

```powershell
python -m scripts.plot_loss --metrics logs/overfit/metrics.json --output logs/overfit/loss_curve.png
```

Generate text samples from a trained checkpoint:

```powershell
python -m scripts.sample_generations \
  --checkpoint logs/overfit/overfit_final.pt \
  --tokenizer logs/overfit/tokenizer.json \
  --output logs/overfit/sample_generations.json
```

## Train Stage 1 Model (Long/GPU)

Orchestrate GPU-optimized training with auto-resume and checkpoint rotation:

```powershell
python scripts/train_long.py --config configs/stage1_tinystories.toml --keep-last-n 5
```

## Monitor Training Live

Continuous real-time telemetry dashboard (GPU, VRAM, Loss, LR, tokens/sec, ETA, latest sample):

```powershell
python scripts/live_dashboard.py
```

## Run Generation Benchmarks

Run quantitative generation benchmarks (n-gram repeats, sentence length, punctuation density, token entropy, coherence) on a checkpoint:

```powershell
python scripts/benchmark_generation.py --checkpoint checkpoints/stage1/latest.pt
```

## Export Inference Checkpoints

Strip training states from a checkpoint to yield a clean model state dict + config JSON:

```powershell
python scripts/export_checkpoint.py --checkpoint checkpoints/stage1/latest.pt
```

## Export Hugging Face Packages

Package weights, tokenizer, config.json, generation_config.json, custom configuration/modeling files, and README cards for future publishing under `NexaraAI/Nexara-0.1`:

```powershell
python scripts/export_huggingface.py --checkpoint checkpoints/stage1/latest.pt --output-dir exports/huggingface/Nexara-0.1
```

## Estimate Model Size

```powershell
python -m scripts.estimate_params --config configs/stage1_tinystories.toml
```

## Evaluate Checkpoints

Evaluate validation loss, perplexity, text samples, and checkpoint round-trip:

```powershell
python -m scripts.evaluate_checkpoint --checkpoint checkpoints/stage1/latest.pt --config configs/stage1_tinystories.toml
```

## Supervised Fine-Tuning (SFT) (Phase 2)

Build custom identity dataset:
```powershell
python scripts/build_nexara_identity_dataset.py
```

Prepare and blend SFT dataset:
```powershell
python scripts/prepare_sft_dataset.py
```

Run 10-example overfit validation on CPU:
```powershell
python scripts/run_sft_overfit.py
```

Run SFT instruction tuning:
```powershell
python scripts/train_sft.py --config configs/stage2_sft.toml
```

Evaluate SFT instruction-tuned checkpoint:
```powershell
python scripts/evaluate_sft.py --checkpoint checkpoints/stage2/step_500.pt --config configs/stage2_sft.toml
```

## Generate Text

```powershell
python -m inference.generate \
  --checkpoint checkpoints/stage1/latest.pt \
  --prompt "Lily found a little red ball"
```

## Chat

```powershell
python -m inference.chat --checkpoint checkpoints/stage1/latest.pt
```

## Pre-trained Weights

Model weights and tokenizer configurations for **Nexara Nano 100M-Chat** are hosted on the Hugging Face Model Hub:

🤗 **[Emperordzd/Nexara-Nano-100M-Chat](https://huggingface.co/Emperordzd/Nexara-Nano-100M-Chat)**

