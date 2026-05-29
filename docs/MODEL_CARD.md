# Model Card — Qwen-COBOL

## Model Description

| Field | Value |
|---|---|
| Base model | Qwen/Qwen3.6-27B-Instruct (or Qwen3-14B fallback) |
| Architecture | Dense transformer, hybrid thinking mode |
| Training stages | CPT → SFT (curriculum) → DPO |
| Adapter method | DoRA (Weight-Decomposed Low-Rank Adaptation) |
| Quantizations | AWQ-4bit (consumer GPU, ~15GB), FP8 (L40S/H100, ~27GB) |

## Intended Use

On-premise COBOL assistant for:
- Code explanation and summarization
- Refactoring (GO TO elimination, modularization)
- Translation (COBOL↔Java)
- Defect detection and debugging
- Code generation from natural language specifications

## Out of Scope

- Production use without human review
- Dialects beyond GnuCOBOL-compatible syntax (IBM Enterprise COBOL, Micro Focus)
- Sensitive or proprietary COBOL codebases (model trained on public data only)

## Training Details

- **CPT**: ~100-200M token validated COBOL corpus, 2-3 epochs, DoRA r=128
- **SFT**: ~40-50k ChatML examples with `<thinking>` traces, curriculum learning, DoRA r=64
- **DPO**: ~3-5k compiler-validated preference pairs, beta=0.1, DoRA r=64
- **Hardware**: Lightning AI L40S 48GB (free tier)

## Evaluation

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## Limitations

- COBOL dialects not covered by GnuCOBOL may produce invalid output
- Programs requiring copybooks not in context may be incomplete
- Long programs (>8192 tokens) are chunked; cross-chunk reasoning requires retrieval
