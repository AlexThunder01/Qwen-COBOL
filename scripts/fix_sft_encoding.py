"""
Fix BPE encoding artifacts (Ġ/Ċ/<|assistant|>) nei 504 esempi teacher_bulk
già su HF Hub. Re-pusha il split corretto.
"""
from __future__ import annotations

import os, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = Path(__file__).resolve().parent.parent / ".env"
os.environ["HF_TOKEN"] = re.search(r"hf_[A-Za-z0-9]{20,}", env.read_bytes().decode("latin-1")).group(0)

from huggingface_hub import hf_hub_download
from datasets import Dataset
import pandas as pd

SFT_REPO = "AlexThunder0/cobol-sft-dataset"
token = os.environ["HF_TOKEN"]


def clean(text: str) -> str:
    # Prima converti gli artefatti BPE (Ċ non è \s, deve essere sostituito prima del regex)
    text = text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
    # Poi rimuovi l'header <|assistant|> che ora è preceduto da \n normali
    text = re.sub(r'^\s*<\|assistant\|>\s*', '', text)
    return text.strip()


path = hf_hub_download(
    repo_id=SFT_REPO,
    filename="data/teacher_bulk-00000-of-00001.parquet",
    repo_type="dataset", token=token, force_download=True,
)
df = pd.read_parquet(path)
print(f"Record originali: {len(df)}")

# Applica pulizia ai messaggi assistant
def fix_row(msgs):  # msgs è direttamente la lista di messaggi
    fixed = []
    for m in msgs:
        if m["role"] == "assistant":
            fixed.append({"role": "assistant", "content": clean(m["content"])})
        else:
            fixed.append(m)
    return fixed

df["messages"] = df["messages"].apply(fix_row)

# Verifica prima/dopo su 2 esempi
for i in [0, 100]:
    ans = df.iloc[i]["messages"][1]["content"]
    print(f"\n--- Esempio {i} (primi 200 char) ---")
    print(ans[:200])

print(f"\nPush su HF Hub …")
from datasets import Features, Value, Sequence
features = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "source": Value("string"),
    "difficulty_score": Value("float64"),
})
records = df.to_dict("records")
Dataset.from_list(records, features=features).push_to_hub(
    SFT_REPO, split="teacher_bulk", private=True, token=token
)
print("Done.")
