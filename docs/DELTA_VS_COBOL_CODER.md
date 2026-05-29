# Delta vs COBOL-Coder

Analysis of what differentiates Qwen-COBOL from COBOL-Coder ([arxiv 2604.03986](https://arxiv.org/html/2604.03986v1)).

## What COBOL-Coder does

- Base: Qwen2.5-Coder (code-pretrained, advantage on syntax out of the box)
- Training: SFT only, automated data augmentation pipeline
- No DPO, no thinking traces, no retrieval
- SOTA: 73.95% compile, 49.33 Pass@1, 34.93 Java→COBOL

## What Qwen-COBOL adds

| Lever | Why it matters | Expected gain |
|---|---|---|
| CPT on real COBOL corpus (100-200M token) | Compensates for Qwen3.6 not being code-pretrained | +10-15pt compile |
| `<thinking>` traces in SFT | Model reasons before answering; no analog in COBOL-Coder | +3-5pt Pass@1 |
| DPO with compiler ground truth | Directly optimizes for valid COBOL output; absent in COBOL-Coder | +2-4pt compile |
| Larger dataset (40-50k vs unknown) | More diverse coverage of COBOL constructs | +2-3pt Pass@1 |
| LazyGraphRAG retrieval at inference | Multi-program context; novel for COBOL tasks | +? (multi-program tasks only) |

## Why Qwen3.6-27B despite not being code-pretrained

COBOL-Coder's base (Qwen2.5-Coder) has a head start on syntax, but:
- Qwen3.6-27B is larger (27B vs ~32B, comparable)
- Qwen3.6 hybrid thinking mode enables genuine multi-step reasoning
- CPT on COBOL-specific corpus should close the syntax gap within W4

## Risk: foundation gap

If CPT is insufficient to close the gap (measure post-CPT eval in W4), options:
1. Increase CPT epochs (3→4, requires more quota)
2. Fallback to Qwen3-14B (3x faster, more CPT epochs in same quota)
3. Consider Qwen2.5-Coder-based alternative

## Honest assessment (to update with real numbers in W7)

TBD — fill in after final eval.
