# Data Card — Qwen-COBOL Training Corpus

## CPT Corpus (`YOUR_HF_USERNAME/cobol-cpt-corpus`)

| Source | Volume (est.) | License | Notes |
|---|---|---|---|
| XMainframe Training Corpus | 236M token | MIT | [FSoft-AI4Code/XMainframe](https://github.com/FSoft-AI4Code/XMainframe) |
| The Stack v2 dedup (COBOL filter) | TBD | Permissive | [bigcode/the-stack-v2-dedup](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) |
| X-COBOL Zenodo | ~182 repos | Open access | [zenodo.org/records/7968845](https://zenodo.org/records/7968845) |
| Community repos | small | Permissive | opensourcecobol, Martinfx/Cobol, openmainframeproject |
| NIST COBOL 85 test suite | 9740+ programs | Public domain | Ships with gnucobol-bin |

**Processing**: column strip → whitespace normalize → GnuCOBOL syntax validation → MinHash LSH dedup (Jaccard 0.7) → difficulty scoring.

## SFT Dataset (`YOUR_HF_USERNAME/cobol-sft-dataset`)

| Split | Examples | Source | License |
|---|---|---|---|
| mainframebench | 7,052 | [Fsoft-AIC/MainframeBench](https://huggingface.co/datasets/Fsoft-AIC/MainframeBench) | MIT |
| gemini_gold | 4,000–6,000 | Gemini 2.5 Flash (generated) | See Google ToS |
| bulk_teacher | 30,000–40,000 | XMainframe-instruct-10.5b or Qwen3.6-27B-Instruct (generated) | Generated output |

**Format**: ChatML with `<thinking>...</thinking>` traces for complex tasks.
**Validation**: all generated examples pass GnuCOBOL compiler check or manual inspection.

## DPO Dataset (`YOUR_HF_USERNAME/cobol-dpo-dataset`)

| Field | Description |
|---|---|
| prompt | User instruction (list of messages) |
| chosen | Compiler-valid response |
| rejected | Variant with deliberate bug (doesn't compile) |

~3,000–5,000 pairs. Chosen/rejected validated with `cobc -fsyntax-only`.
