"""
Valida gli esempi generate-from-spec compilandoli con GnuCOBOL.

Filtra lo split `generate_spec` (candidati grezzi dal teacher) tenendo solo
gli esempi il cui programma COBOL **compila davvero** → scrive `generate_spec_valid`.

Elimina la spazzatura sintattica: il teacher inventa problema E soluzione senza
verifica, quindi alcuni programmi hanno errori. Solo i compilanti diventano gold.

Gira su Modal CPU (no GPU, ~$0.10). Split separato → nessuna race con la
generazione locale ancora in corso.

Usage:
  python -m modal run scripts/modal_validate_spec.py
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
    .pip_install("datasets", "huggingface-hub")
)
app = modal.App("qwen-cobol-validate-spec", image=image)

SFT_REPO = "AlexThunder0/cobol-sft-dataset"
SRC_SPLIT = "generate_spec"
DST_SPLIT = "generate_spec_valid"
FAIL_SPLIT = "generate_spec_failed"  # non-compilanti + errore cobc (diagnostica + auto-fix)


def _extract_program(assistant: str) -> str | None:
    m = re.search(r"```(?:cobol)?\s*\n?(.*?)```", assistant, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


@app.function(timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")])
def validate() -> dict:
    import os
    from datasets import load_dataset, Dataset, Features, Value

    token = os.environ["HF_TOKEN"]
    ds = load_dataset(SFT_REPO, split=SRC_SPLIT, token=token)
    print(f"Candidati da validare: {len(ds)}")

    cobc_ver = subprocess.run(["cobc", "--version"], capture_output=True, text=True)
    print(f"GnuCOBOL: {cobc_ver.stdout.splitlines()[0] if cobc_ver.stdout else 'NOT FOUND'}")

    def try_compile(code: str) -> tuple[bool, str]:
        """Ritorna (compila, stderr). stderr utile per diagnostica/auto-fix."""
        with tempfile.NamedTemporaryFile(suffix=".cob", mode="w", delete=False, dir="/tmp") as f:
            f.write(code)
            p = f.name
        try:
            cp = subprocess.run(
                ["cobc", "-w", "-fformat=variable", "-c", p],
                capture_output=True, text=True, timeout=15, cwd="/tmp",
            )
            ok, err = cp.returncode == 0, cp.stderr
        except Exception as e:
            ok, err = False, str(e)
        Path(p).unlink(missing_ok=True)
        for ext in (".o", ".i"):
            Path(p).with_suffix(ext).unlink(missing_ok=True)
        return ok, err

    kept = []
    failed = []
    n_ok = n_bad = n_noprog = 0
    for row in ds:
        prog = _extract_program(row["messages"][1]["content"])
        if not prog:
            n_noprog += 1
            continue
        ok, err = try_compile(prog)
        if ok:
            kept.append({
                "messages": list(row["messages"]),
                "source": row["source"],
                "difficulty_score": float(row["difficulty_score"]),
            })
            n_ok += 1
        else:
            failed.append({
                "messages": list(row["messages"]),
                "source": row["source"],
                "difficulty_score": float(row["difficulty_score"]),
                "compile_error": err[:2000],
            })
            n_bad += 1

    print(f"\n{'='*50}")
    print(f"Compilano:        {n_ok}")
    print(f"Non compilano:    {n_bad}")
    print(f"Senza programma:  {n_noprog}")
    print(f"Tasso validità:   {100*n_ok/max(len(ds),1):.1f}%")
    print(f"{'='*50}")

    sft_feats = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "source": Value("string"),
        "difficulty_score": Value("float64"),
    })

    if kept:
        Dataset.from_list(kept, features=sft_feats).push_to_hub(
            SFT_REPO, split=DST_SPLIT, private=True, token=token
        )
        print(f"Pushati {n_ok} validati → {DST_SPLIT}")

    if failed:
        fail_feats = Features({
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "source": Value("string"),
            "difficulty_score": Value("float64"),
            "compile_error": Value("string"),
        })
        Dataset.from_list(failed, features=fail_feats).push_to_hub(
            SFT_REPO, split=FAIL_SPLIT, private=True, token=token
        )
        print(f"Pushati {n_bad} falliti (con errore) → {FAIL_SPLIT}")

        # Sintesi pattern di errore più comuni (prima riga utile di cobc)
        from collections import Counter
        patterns = Counter()
        for f in failed:
            for line in f["compile_error"].splitlines():
                m = re.search(r"error:\s*(.*)", line, re.IGNORECASE)
                if m:
                    # normalizza nomi variabili/numeri per raggruppare
                    msg = re.sub(r"'[^']*'", "'X'", m.group(1))
                    msg = re.sub(r"\d+", "N", msg)
                    patterns[msg.strip()[:80]] += 1
                    break
        print("\n=== TOP 10 PATTERN DI ERRORE ===")
        for msg, cnt in patterns.most_common(10):
            print(f"  {cnt:3d}x  {msg}")

    return {"ok": n_ok, "bad": n_bad, "noprog": n_noprog}


@app.local_entrypoint()
def main():
    result = validate.remote()
    print(f"Completato: {result}")
