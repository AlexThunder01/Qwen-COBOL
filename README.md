# Qwen-COBOL — Enterprise COBOL Assistant

Specialized LLM for analysis, explanation, refactoring, and translation of legacy COBOL code.
Built on **Qwen3.6-27B-Instruct** with CPT → SFT (thinking traces) → DPO pipeline.

**Target**: beat [COBOL-Coder](https://arxiv.org/html/2604.03986v1) on COBOLEval (>75% compile, >50 Pass@1) and COBOL-JavaTrans (>36 Pass@1).

## Quick Start

See [PREREQUISITES.md](PREREQUISITES.md) for WSL2, GnuCOBOL, and account setup.

```bash
# WSL2 Ubuntu 24.04 — all training/eval commands run here
pip install uv
uv sync
```

## Pipeline overview

```
CPT (W4)  →  SFT + curriculum (W5)  →  DPO (W6)  →  AWQ/FP8 quantize (W7)
  ↓                  ↑
Corpus W2      Synth data W3
```

| Stage | Script | Platform |
|---|---|---|
| Data ingest | `src/pipeline/ingest.py` | Kaggle T4 |
| Synthetic data | `src/synth/distill_orchestrator.py` | Lightning L40S / Kaggle |
| CPT | `src/train/cpt.py` | Lightning L40S |
| SFT | `src/train/sft.py` | Lightning L40S |
| DPO | `src/train/dpo.py` | Lightning L40S |
| Eval | `src/eval/cobolceval_runner.py` | Kaggle T4 |
| Deploy | `src/deploy/vllm_serve.sh` | Lightning L40S / local Docker |

## Benchmarks

| Model | COBOLEval Compile | COBOLEval Pass@1 | Java→COBOL Pass@1 |
|---|---|---|---|
| GPT-4o | 41.8% | 16.4 | ~0 |
| COBOL-Coder (SOTA) | 73.95% | 49.33 | 34.93 |
| **Qwen-COBOL (target)** | **>75%** | **>50** | **>36** |

## Hardware (free tier only)

| Resource | Spec | Use |
|---|---|---|
| Lightning AI Studio | L40S 48GB | Training CPT/SFT/DPO, teacher inference |
| Kaggle | 2x T4 16GB, 30h/week | Data prep, eval, teacher inference (secondary) |
| Google AI Studio | Gemini 2.5 Flash, 1500 req/day | Gold SFT examples |
| HF Hub private | ~50GB LFS | Raw corpus + DoRA checkpoints |

## Repository structure

```
config/          training YAML configs
src/
  pipeline/      data ingest, parse, clean, dedupe, validate
  synth/         synthetic data generation (Gemini + teacher vLLM)
  train/         CPT, SFT, DPO scripts
  retrieval/     LazyGraphRAG + Neo4j + Qdrant
  eval/          COBOLEval, COBOL-JavaTrans runners
  deploy/        merge DoRA, quantize AWQ/FP8, vLLM serve
notebooks/       Kaggle + Lightning notebooks
docker/          Neo4j, Qdrant, vLLM compose files
docs/            DATA_CARD, MODEL_CARD, BENCHMARK_RESULTS
data/            gitignored — lives on HF Hub
```
