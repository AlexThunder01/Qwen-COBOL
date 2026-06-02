"""
Cleanup del corpus CPT su parquet esistente (no re-download da S3).

Filtro raffinato (vedi analisi inspect_corpus.py):
- KEEP se ha marker COBOL: DIVISION | PIC | keyword proc | SELECT/ASSIGN/FILE-CONTROL
  | numeri di livello (01-49 NOME) — evita falsi negativi sui copybook
- DROP markup vero (>20 tag <...)
- DROP solo encoding davvero rotto (>40% non-ascii) — rilassato per tenere COBOL
  con commenti CJK (giapponese ORCA, cinese)
- TRUNCATE a 64k char per limitare la dominanza dei file giganti

Re-push su AlexThunder0/cobol-cpt-corpus (overwrite).
"""
from __future__ import annotations

import os, re, sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from datasets import Dataset
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = Path(__file__).resolve().parent.parent / ".env"
os.environ["HF_TOKEN"] = re.search(r"hf_[A-Za-z0-9]{20,}", env.read_bytes().decode("latin-1")).group(0)

REPO = "AlexThunder0/cobol-cpt-corpus"
MAX_CHARS = 64_000

path = hf_hub_download(repo_id=REPO, filename="data/train-00000-of-00001.parquet",
                       repo_type="dataset", token=os.environ["HF_TOKEN"])
df = pd.read_parquet(path)
df["len"] = df["content"].str.len()
n0 = len(df)
tok0 = df["len"].sum() / 4
C = df["content"]

def has(pat):
    return C.str.contains(pat, case=False, regex=True, na=False)

# ── Marker COBOL allargati ────────────────────────────────────────────────────
m_div = has(r"\b(IDENTIFICATION|DATA|PROCEDURE|ENVIRONMENT)\s+DIVISION\b")
m_pic = has(r"\bPIC(TURE)?\s+[X9SVAP(]")
m_kw  = has(r"\b(PERFORM|MOVE|WORKING-STORAGE|COMPUTE|DISPLAY|GOBACK|STOP\s+RUN|EVALUATE)\b")
m_sel = has(r"\b(SELECT\s+\w+\s+ASSIGN|FILE-CONTROL|ASSIGN\s+TO|ORGANIZATION\s+IS)\b")
m_lvl = has(r"(?m)^\s*\d{2}\s+[A-Z][\w-]+\s+(PIC|OCCURS|REDEFINES|VALUE|COMP)")
cobol_ok = m_div | m_pic | m_kw | m_sel | m_lvl

# ── Scarti ────────────────────────────────────────────────────────────────────
m_markup = C.str.count(r"<[a-zA-Z/!]") > 20

def nonascii_ratio(s):
    if not s:
        return 0.0
    chunk = s[:2000]
    return sum(1 for ch in chunk if ord(ch) > 126) / len(chunk)
df["nar"] = C.map(nonascii_ratio)
m_binary = df["nar"] > 0.40  # rilassato: tiene COBOL con commenti CJK

keep = cobol_ok & ~m_markup & ~m_binary
removed = (~keep).sum()

print("=== CLEANUP ===")
print(f"  Record iniziali:   {n0:,}  (~{tok0/1e6:.1f}M token)")
print(f"  Rimossi:           {removed:,}")
print(f"    no marker COBOL: {(~cobol_ok).sum():,}")
print(f"    markup (>20 tag):{m_markup.sum():,}")
print(f"    binary (>40%):   {m_binary.sum():,}")

out = df[keep].copy()
n_capped = (out["len"] > MAX_CHARS).sum()
out["content"] = out["content"].str.slice(0, MAX_CHARS)
out["len"] = out["content"].str.len()
tok1 = out["len"].sum() / 4

print(f"  Record tenuti:     {len(out):,}")
print(f"  File troncati 64k: {n_capped:,}")
print(f"  Token finali:      ~{tok1/1e6:.1f}M")
print(f"\n  Per fonte:")
print(out["source"].value_counts().to_string())

# Ricostruisci difficulty_score? lo teniamo com'era (gia' presente).
cols = ["content", "source", "path", "difficulty_score"]
out = out[[c for c in cols if c in out.columns]].reset_index(drop=True)

print(f"\n  Push su {REPO} …")
ds = Dataset.from_pandas(out)
ds.push_to_hub(REPO, split="train", private=True, token=os.environ["HF_TOKEN"])
print("  Fatto.")
