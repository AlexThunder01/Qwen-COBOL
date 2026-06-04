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
PUSH_EVERY = 50

# Teacher frontier in riserva — tutti battono lo student Qwen3.6-27B
TEACHER_MODELS = [
    "qwen3.7-max-preview",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3-max",
    "qwen3.6-max-preview",
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
    "Write code valid for GnuCOBOL 3.x in free-ish fixed format (area B indentation)."
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
    code_m = re.search(r"```(?:cobol)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
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


def main(target: int) -> None:
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
    consecutive_failures = 0
    random.seed(13)

    while len(accumulated) < target:
        model = TEACHER_MODELS[model_idx % len(TEACHER_MODELS)]
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
                temperature=0.85,  # alta per diversità dei task
                **extra,
            )
            text = resp.choices[0].message.content or ""
            consecutive_failures = 0
        except Exception as e:
            msg = str(e)
            if "FreeTierOnly" in msg or "exhausted" in msg or "429" in msg or "403" in msg:
                consecutive_failures += 1
                if consecutive_failures >= 8:
                    logger.warning("Modello %s esaurito → passo al prossimo", model)
                    model_idx += 1
                    consecutive_failures = 0
                    if model_idx >= len(TEACHER_MODELS):
                        logger.warning("Tutti i modelli esauriti — stop")
                        break
                continue
            logger.warning("Errore (%s): %s", model, msg[:120])
            time.sleep(1)
            continue

        parsed = parse_response(text)
        if not parsed:
            continue
        approach, program = parsed
        ex = build_example(approach, program, model)
        if not ex:
            continue

        accumulated.append(ex)
        new_since_push += 1
        time.sleep(0.5)

        if new_since_push >= PUSH_EVERY:
            push(accumulated)
            logger.info("Checkpoint: %d/%d su HF Hub (model corrente: %s)", len(accumulated), target, model)
            new_since_push = 0

    push(accumulated)
    logger.info("Generate-from-spec completato: %d esempi", len(accumulated))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1500)
    args = p.parse_args()
    main(args.target)
