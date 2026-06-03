"""
Ispezione qualità campione del dataset SFT (teacher_bulk split).
Mostra esempi random per task type + statistiche aggregate.
"""
from __future__ import annotations

import os, re, sys, random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = Path(__file__).resolve().parent.parent / ".env"
os.environ["HF_TOKEN"] = re.search(r"hf_[A-Za-z0-9]{20,}", env.read_bytes().decode("latin-1")).group(0)

from huggingface_hub import hf_hub_download
import pandas as pd

SFT_REPO = "AlexThunder0/cobol-sft-dataset"

path = hf_hub_download(
    repo_id=SFT_REPO,
    filename="data/teacher_bulk-00000-of-00001.parquet",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    force_download=True,
)
df = pd.read_parquet(path)
print(f"Totale esempi teacher_bulk: {len(df)}\n")

# Statistiche aggregate
def msg(row, role): return next((m["content"] for m in row["messages"] if m["role"] == role), "")
df["user_len"]   = df["messages"].apply(lambda m: len(next((x["content"] for x in m if x["role"]=="user"), "")))
df["answer_len"] = df["messages"].apply(lambda m: len(next((x["content"] for x in m if x["role"]=="assistant"), "")))
df["task"] = df["source"].str.extract(r"_([^_]+)$")

print("=== STATISTICHE ===")
print(f"Lunghezza risposta: median={df['answer_len'].median():,.0f}  min={df['answer_len'].min()}  max={df['answer_len'].max():,}")
print(f"Risposte vuote: {(df['answer_len']==0).sum()}")
print(f"\nPer task:")
print(df.groupby("task")["answer_len"].agg(["count","median"]).to_string())

# Campione per task (2 esempi per tipo)
print("\n" + "="*70)
print("CAMPIONE QUALITA' (1 esempio per task)")
print("="*70)
random.seed(42)
for task in df["task"].dropna().unique():
    subset = df[df["task"]==task]
    row = subset.sample(1, random_state=42).iloc[0]
    user = row["messages"][0]["content"]
    answer = row["messages"][1]["content"]
    print(f"\n{'─'*60}")
    print(f"TASK: {task}  |  diff={row['difficulty_score']:.2f}  |  answer_len={len(answer)}")
    print(f"\n[USER — primi 300 char]\n{user[:300]}")
    print(f"\n[ASSISTANT — primi 600 char]\n{answer[:600]}")
