"""
Auto-fix dei generate-from-spec falliti: rimanda ogni programma + errore cobc al
teacher Alibaba per la correzione, ri-compila, e i recuperati confluiscono in
generate_spec_valid.

Loop su Modal (Alibaba API via secret + GnuCOBOL). Recupera errori di formato
residui E errori di codice veri (il teacher corregge guidato dall'errore esatto).

Usage:
  python -m modal run scripts/modal_autofix_spec.py
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("gnucobol")
    .pip_install("datasets", "huggingface-hub", "openai")
)
app = modal.App("qwen-cobol-autofix-spec", image=image)

SFT_REPO = "AlexThunder0/cobol-sft-dataset"
FAIL_REPO = "AlexThunder0/cobol-spec-failed"
VALID_SPLIT = "generate_spec_valid"
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
FIX_MODELS = ["qwen3-coder-plus", "qwen3.6-plus", "qwen3-max", "qwen3.7-plus"]

FIX_PROMPT = """\
The following COBOL subprogram fails to compile with GnuCOBOL 3.x (fixed/variable format,
indicator in column 7, code in area A/B from column 8).

Compile error:
{error}

Program:
```cobol
{program}
```

Fix ONLY what's needed to make it compile (keep the same logic, PROGRAM-ID, and LINKAGE
interface). Comments must have `*` in column 7. Output ONLY the corrected complete program
inside a single ```cobol code block, nothing else."""


def _reindent(prog: str) -> str:
    out = []
    for line in prog.split("\n"):
        if not line.strip():
            out.append(line)
        elif line[:1] in (" ", "\t"):
            out.append(line)
        elif line.lstrip().startswith("*"):
            out.append("      " + line)
        else:
            out.append("       " + line)
    return "\n".join(out)


def _extract(text: str) -> str | None:
    m = re.search(r"```(?:cobol)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip("\n") if m else None


@app.function(
    timeout=2 * 3600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("alibaba-secret"),
    ],
)
def autofix() -> dict:
    import os
    from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
    from openai import OpenAI

    token = os.environ["HF_TOKEN"]
    client = OpenAI(api_key=os.environ["ALIBABA_API"], base_url=DASHSCOPE_BASE_URL)

    failed = load_dataset(FAIL_REPO, split="train", token=token)
    print(f"Falliti da auto-fixare: {len(failed)}")

    def compiles(code: str) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".cob", mode="w", delete=False, dir="/tmp") as f:
            f.write(code); p = f.name
        try:
            cp = subprocess.run(["cobc", "-w", "-fformat=variable", "-c", p],
                                capture_output=True, text=True, timeout=15, cwd="/tmp")
            ok = cp.returncode == 0
        except Exception:
            ok = False
        Path(p).unlink(missing_ok=True)
        return ok

    recovered = []
    model_idx = 0
    consec_quota = 0
    for i, row in enumerate(failed):
        prog = _extract(row["messages"][1]["content"])
        if not prog:
            continue
        model = FIX_MODELS[model_idx % len(FIX_MODELS)]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": FIX_PROMPT.format(
                    error=row["compile_error"][:500], program=prog)}],
                max_tokens=2048, temperature=0.2,
            )
            fixed = _extract(resp.choices[0].message.content or "")
            consec_quota = 0
        except Exception as e:
            if any(k in str(e) for k in ("FreeTierOnly", "exhausted", "429", "403")):
                consec_quota += 1
                if consec_quota >= 6:
                    model_idx += 1
                    consec_quota = 0
                    if model_idx >= len(FIX_MODELS):
                        print("Tutti i modelli esauriti")
                        break
            continue

        if not fixed:
            continue
        fixed = _reindent(fixed)
        if compiles(fixed):
            user_msg = row["messages"][0]["content"]
            recovered.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"```cobol\n{fixed}\n```"},
                ],
                "source": row["source"] + "_autofixed",
                "difficulty_score": float(row["difficulty_score"]),
            })
        if (i + 1) % 30 == 0:
            print(f"Processati {i+1}/{len(failed)}, recuperati {len(recovered)}")

    print(f"\nRecuperati via auto-fix: {len(recovered)}/{len(failed)}")

    if recovered:
        feats = Features({
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "source": Value("string"), "difficulty_score": Value("float64"),
        })
        # Merge con i validi esistenti
        existing = load_dataset(SFT_REPO, split=VALID_SPLIT, token=token)
        existing = existing.cast(feats) if existing.features != feats else existing
        merged = concatenate_datasets([existing, Dataset.from_list(recovered, features=feats)])
        merged.push_to_hub(SFT_REPO, split=VALID_SPLIT, private=True, token=token)
        print(f"generate_spec_valid: {len(existing)} → {len(merged)} (+{len(recovered)})")
    return {"recovered": len(recovered), "total_failed": len(failed)}


@app.local_entrypoint()
def main():
    print(autofix.remote())
