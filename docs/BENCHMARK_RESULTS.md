# Benchmark Results

Results updated after each training stage (W4–W7).

## COBOLEval (146 problems)

| Stage | Compile Rate | Pass@1 | Notes |
|---|---|---|---|
| GPT-4o (reference) | 41.8% | 16.4% | From COBOL-Coder paper |
| COBOL-Coder (SOTA) | 73.95% | 49.33% | Qwen2.5-Coder + SFT |
| Qwen3.6-27B vanilla | TBD | TBD | W1 baseline |
| + CPT | TBD | TBD | W4 |
| + SFT | TBD | TBD | W5 |
| + DPO | TBD | TBD | W6 |
| + DPO + retrieval | TBD | TBD | W6 |
| **Target stretch** | **>75%** | **>50%** | Beats COBOL-Coder |

## COBOL-JavaTrans (143 pairs)

| Stage | Pass@1 | Notes |
|---|---|---|
| COBOL-Coder (SOTA) | 34.93% | Qwen2.5-Coder + SFT |
| **Qwen-COBOL final** | TBD | |
| **Target stretch** | **>36%** | |

## Ablation Study

| Config | COBOLEval Compile | COBOLEval Pass@1 |
|---|---|---|
| Base vanilla | TBD | TBD |
| + CPT only | TBD | TBD |
| + CPT + SFT | TBD | TBD |
| + CPT + SFT + DPO | TBD | TBD |
| + CPT + SFT + DPO + retrieval | TBD | TBD |

## Notes on methodology

- All evals run with temperature=0.0 (greedy decoding), thinking_budget=1024 tokens.
- Compile rate measured with `cobc -fsyntax-only` (GnuCOBOL 3.2).
- Pass@1 requires all test cases in the COBOLEval harness to pass.
