"""
Genera esempi SFT `generate-from-spec` allineati a COBOLEval.

Task target del benchmark: dato uno scheletro COBOL (IDENTIFICATION + DATA +
LINKAGE con la firma + commento-specifica), completare la PROCEDURE DIVISION.
Questo task è quasi assente dal dataset attuale (explain/refactor/debug) → gap.

Strategia:
  - Teacher frontier Alibaba (qwen3.7-max ecc, battono lo student Qwen3.6-27B)
  - Per ogni chiamata: il teacher inventa UN esercizio COBOL in stile benchmark
    (sottoprogramma con LINKAGE) + approccio conciso + programma completo
  - Costruiamo la coppia SFT: user = scheletro+istruzione (formato eval),
    assistant = approccio breve + programma completo in ```cobol
  - NIENTE problemi reali di COBOLEval come seed → zero contaminazione del test

Resume da HF Hub, push ogni 50 esempi. Gira local (CPU) o Kaggle.
Needs: ALIBABA_API, HF_TOKEN.

Usage:
  python scripts/gen_spec_data.py --target 1500
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Carica .env (encoding-robusto)
env = Path(__file__).resolve().parent.parent / ".env"
raw = env.read_bytes().decode("latin-1")
for key in ("ALIBABA_API", "HF_TOKEN"):
    m = re.search(rf"{key}\s*=\s*(\S+)", raw)
    if m:
        os.environ[key] = m.group(1).strip()
# HF token a volte è hf_... → estrai pulito
m = re.search(r"hf_[A-Za-z0-9]{20,}", raw)
if m:
    os.environ["HF_TOKEN"] = m.group(0)

from openai import OpenAI
from datasets import Dataset, load_dataset, Features, Value
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
SFT_REPO = "AlexThunder0/cobol-sft-dataset"
SPLIT = "generate_spec"
PUSH_EVERY = 20      # salva prima
CONCURRENCY = 5      # chiamate parallele per batch

# Teacher frontier con quote separate — tutti ≥ student Qwen3.6-27B.
# Diversità di famiglia (DeepSeek + GLM oltre a Qwen) → dati SFT più robusti.
TEACHER_MODELS = [
    "deepseek-v4-pro",              # frontier coder, famiglia diversa
    "qwen3-coder-next",             # code specialist Qwen
    "glm-5.1",                      # frontier GLM, famiglia diversa
    "qwen3.7-max",                  # flagship GA
    "qwen3.7-max-2026-05-20",       # snapshot (quota separata)
    "qwen3.7-max-2026-05-17",       # snapshot
    "qwen3-coder-plus-2025-09-23",  # coder snapshot
    "qwen3.7-plus-2026-05-26",      # plus snapshot
]

# Domini algoritmici per diversità (NO problemi COBOLEval reali)
TOPICS = [
    "string manipulation (reverse, count chars, find substring)",
    "numeric computation (factorial, GCD, prime check, power)",
    "array processing (sum, max, min, average, count matching)",
    "searching and sorting a small fixed array",
    "data validation (check format, range, parity)",
    "financial calculation (interest, discount, tax, totals)",
    "date and time logic (day of week, days between, leap year)",
    "character encoding and case conversion",
    "boolean logic and conditional flags",
    "counting and frequency analysis",
    "rounding, truncation, and decimal handling",
    "pattern checking (palindrome, sorted, balanced)",
]

DIFFICULTIES = ["easy", "easy", "medium", "medium", "hard"]

SYSTEM_PROMPT = (
    "You are an expert COBOL educator who creates original programming exercises "
    "in the style of coding benchmarks (HumanEval-like, but for COBOL subprograms). "
    "Write code that COMPILES with GnuCOBOL 3.x in FIXED format.\n\n"
    "CRITICAL format & syntax rules (the most common compile errors to avoid):\n"
    "- FIXED format: comment lines have '*' in column 7 (six spaces then '*'); "
    "all code in area B (indent ~7 spaces). NEVER start a line at column 1.\n"
    "- Keep every line <= 72 characters. For long statements, split across lines "
    "(area B), do NOT exceed the margin.\n"
    "- Intrinsic functions: use ONLY valid GnuCOBOL ones (FUNCTION INTEGER, FUNCTION MOD, "
    "FUNCTION NUMVAL, FUNCTION REVERSE, FUNCTION UPPER-CASE...). NEVER 'FUNCTION INT'.\n"
    "- PERFORM VARYING: FROM and BY take a single identifier or literal, NOT an "
    "expression. Compute 'WS-I + 1' into a variable first if needed.\n"
    "- Reference OCCURS/array items WITH a subscript, e.g. WS-ARR(WS-I).\n"
    "- End the program with GOBACK (it is a subprogram), then 'END PROGRAM name.'.\n"
    "- Boolean conditions: use IF ... AND/OR ... with full relational expressions."
)

USER_TEMPLATE = """\
Create ONE original COBOL programming exercise on the topic of: {topic}
Difficulty: {difficulty}

Requirements for the COBOL program:
- It must be a SUBPROGRAM that receives inputs and returns a result via the LINKAGE SECTION.
- Include IDENTIFICATION DIVISION, DATA DIVISION with a LINKAGE SECTION declaring the
  input fields and a field named RESULT for the output, and PROCEDURE DIVISION USING
  the linkage items.
- Right before the LINKAGE SECTION, add a COBOL comment block (lines starting with `*`)
  that clearly describes WHAT the program must compute (the specification), with a small example.
- Make it self-contained and compilable with GnuCOBOL 3.x. Use GOBACK to return (NOT STOP RUN).
- Do NOT copy any known benchmark problem; invent a fresh task.

Output EXACTLY in this format, nothing else:
[APPROACH]
<3 to 5 sentences describing the approach, concise>
[PROGRAM]
```cobol
<the complete COBOL program>
```"""


def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["ALIBABA_API"], base_url=DASHSCOPE_BASE_URL)


def parse_response(text: str) -> tuple[str, str] | None:
    """Estrae (approach, full_program) dalla risposta del teacher."""
    approach_m = re.search(r"\[APPROACH\](.*?)\[PROGRAM\]", text, re.DOTALL | re.IGNORECASE)
    # [ \t]*\r?\n preserva l'indentazione della 1ª riga (NON usare \s* goloso)
    code_m = re.search(r"```(?:cobol)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not code_m:
        return None
    program = code_m.group(1).strip()
    approach = approach_m.group(1).strip() if approach_m else ""
    # Validazione minima: deve avere le sezioni chiave
    if not re.search(r"IDENTIFICATION\s+DIVISION", program, re.IGNORECASE):
        return None
    if not re.search(r"PROCEDURE\s+DIVISION", program, re.IGNORECASE):
        return None
    return approach, program


def make_skeleton(program: str) -> str | None:
    """Rimuove il corpo della PROCEDURE DIVISION → scheletro da completare."""
    m = re.search(r"(.*?PROCEDURE\s+DIVISION[^\n]*\n)", program, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).rstrip()


# Istruzione IDENTICA a quella usata in modal_eval_sft.py (coerenza train/test)
INSTRUCTION = (
    "Complete the following COBOL program by implementing the PROCEDURE DIVISION.\n"
    "Reason step by step about the logic, then provide the complete, compilable "
    "COBOL program inside a single ```cobol code block at the end.\n\n"
    "```cobol\n{prompt}\n```"
)


def build_example(approach: str, program: str, model: str) -> dict | None:
    skeleton = make_skeleton(program)
    if not skeleton:
        return None
    user = INSTRUCTION.format(prompt=skeleton)
    # Target: approccio conciso (insegna a ragionare BREVE) + programma completo
    assistant = f"{approach}\n\n```cobol\n{program}\n```" if approach else f"```cobol\n{program}\n```"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": f"genspec_{model.replace('-', '_').replace('.', '_')}",
        "difficulty_score": 0.6,
    }


def push(examples: list[dict]) -> None:
    feats = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "source": Value("string"),
        "difficulty_score": Value("float64"),
    })
    Dataset.from_list(examples, features=feats).push_to_hub(
        SFT_REPO, split=SPLIT, private=True, token=os.environ["HF_TOKEN"]
    )


def gen_one(client, model: str) -> tuple[dict | None, str]:
    """Genera UN esempio. Ritorna (esempio|None, esito) per la cascata.
    esito ∈ {"ok", "quota", "parse", "other"}."""
    topic = random.choice(TOPICS)
    difficulty = random.choice(DIFFICULTIES)
    user = USER_TEMPLATE.format(topic=topic, difficulty=difficulty)
    try:
        extra = {}
        if "thinking" in model or "qwq" in model:
            extra = {"extra_body": {"enable_thinking": True}}
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            temperature=0.85,
            **extra,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        msg = str(e)
        if any(k in msg for k in ("FreeTierOnly", "exhausted", "429", "403")):
            return None, "quota"
        return None, "other"

    parsed = parse_response(text)
    if not parsed:
        return None, "parse"
    approach, program = parsed
    ex = build_example(approach, program, model)
    return (ex, "ok") if ex else (None, "parse")


def main(target: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    client = get_client()

    # Resume da HF
    accumulated: list[dict] = []
    try:
        existing = load_dataset(SFT_REPO, split=SPLIT, token=os.environ["HF_TOKEN"])
        accumulated = [dict(r) for r in existing]
        logger.info("Ripreso: %d esempi già presenti", len(accumulated))
    except Exception:
        logger.info("Nessun %s su HF — parto da zero", SPLIT)

    if len(accumulated) >= target:
        logger.info("Target già raggiunto (%d) — esco", len(accumulated))
        return

    new_since_push = 0
    model_idx = 0
    random.seed(13)

    # Batch concorrenti: CONCURRENCY chiamate parallele, poi push se serve.
    # Tra un batch e l'altro nessun thread attivo → push sicuro, no lock.
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        while len(accumulated) < target and model_idx < len(TEACHER_MODELS):
            model = TEACHER_MODELS[model_idx]
            futures = [pool.submit(gen_one, client, model) for _ in range(CONCURRENCY)]

            quota_hits = 0
            for fut in futures:
                ex, esito = fut.result()
                if ex:
                    accumulated.append(ex)
                    new_since_push += 1
                elif esito == "quota":
                    quota_hits += 1

            if new_since_push >= PUSH_EVERY:
                push(accumulated)
                logger.info("Checkpoint: %d/%d su HF (model: %s)", len(accumulated), target, model)
                new_since_push = 0

            # Intero batch in quota → modello esaurito, passa al prossimo
            if quota_hits >= CONCURRENCY:
                logger.warning("Modello %s esaurito → prossimo", model)
                model_idx += 1

    push(accumulated)
    if model_idx >= len(TEACHER_MODELS):
        logger.warning("Tutti i modelli esauriti.")
    logger.info("Generate-from-spec completato: %d esempi", len(accumulated))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1500)
    args = p.parse_args()
    main(args.target)
