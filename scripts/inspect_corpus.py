"""
Ultima analisi PRE-CLEANUP: valida che il filtro non scarti COBOL buono.
Ispeziona cosa verrebbe RIMOSSO + cluster header ripetuti + record corti + .ccp.
"""
from __future__ import annotations

import os, re, sys
from pathlib import Path
from collections import Counter

from huggingface_hub import hf_hub_download
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = Path(__file__).resolve().parent.parent / ".env"
os.environ["HF_TOKEN"] = re.search(r"hf_[A-Za-z0-9]{20,}", env.read_bytes().decode("latin-1")).group(0)

path = hf_hub_download(repo_id="AlexThunder0/cobol-cpt-corpus",
    filename="data/train-00000-of-00001.parquet", repo_type="dataset", token=os.environ["HF_TOKEN"])
df = pd.read_parquet(path)
df["len"] = df["content"].str.len()
C = df["content"]
def has(pat): return C.str.contains(pat, case=False, regex=True, na=False)

def clip(s, k=180):
    return s[:k].replace("\n", " \\n ").encode("ascii", "replace").decode("ascii")

# Filtro proposto
m_div = has(r"\b(IDENTIFICATION|DATA|PROCEDURE|ENVIRONMENT)\s+DIVISION\b")
m_pic = has(r"\bPIC(TURE)?\s+[X9SVAP(]")
m_kw  = has(r"\b(PERFORM|MOVE|WORKING-STORAGE|COMPUTE|DISPLAY|GOBACK|STOP\s+RUN)\b")
cobol_ish = m_div | m_pic | m_kw
m_markup = C.str.count(r"<[a-zA-Z/!]") > 20
def nar(s):
    if not s: return 0.0
    return sum(1 for ch in s[:2000] if ord(ch) > 126) / min(len(s), 2000)
df["nar"] = C.map(nar)
m_binary = df["nar"] > 0.15
dirty = (~cobol_ish) | m_markup | m_binary

# ── 1. COSA VIENE RIMOSSO: campione dei record scartati ─────────────────────
print(f"=== RECORD CHE VERREBBERO RIMOSSI: {dirty.sum():,} ===")
removed = df[dirty].copy()
print("\n-- 12 scartati random (per capire se sono davvero non-COBOL) --")
for _, r in removed.sample(min(12, len(removed)), random_state=1).iterrows():
    reason = []
    if not cobol_ish[_]: reason.append("no-cobol-marker")
    if m_markup[_]: reason.append("markup")
    if m_binary[_]: reason.append(f"binary({df['nar'][_]:.0%})")
    print(f"\n[{r['len']:>8,}c {','.join(reason)}] {str(r['path'])[:50]}")
    print(f"   {clip(r['content'])}")

# ── 2. FALSI POSITIVI? record scartati ma che SEMBRANO cobol (hanno .cbl/.cob)
print("\n\n=== POSSIBILI FALSI POSITIVI (scartati ma ext .cbl/.cob/.cpy) ===")
df["ext"] = df["path"].map(lambda p: (re.search(r"\.([a-z0-9]+)$", str(p).lower()) or [None,"?"])[1] if isinstance(re.search(r"\.([a-z0-9]+)$", str(p).lower()), re.Match) else "?")
fp = df[dirty & df["ext"].isin(["cbl","cob","cpy","cobol"])]
print(f"  {len(fp):,} record con estensione COBOL ma classificati sporchi")
for _, r in fp.sample(min(8, len(fp)), random_state=2).iterrows():
    print(f"\n[{r['len']:>7,}c nar={df['nar'][_]:.0%}] {str(r['path'])[:50]}")
    print(f"   {clip(r['content'])}")

# ── 3. Record CORTI (vicino al min 49) ───────────────────────────────────────
print("\n\n=== RECORD CORTI (< 150 char): "
      f"{(df['len']<150).sum():,} ===")
for _, r in df[df["len"]<150].head(8).iterrows():
    print(f"  [{r['len']:>4}c] {clip(r['content'],120)}")

# ── 4. Header ripetuti: i cluster piu' grandi ────────────────────────────────
df["head"] = C.str[:300]
print("\n\n=== CLUSTER HEADER RIPETUTI (top 8) ===")
vc = df["head"].value_counts()
for head, cnt in vc.head(8).items():
    if cnt > 1:
        print(f"\n  x{cnt} record stesso inizio:")
        print(f"   {clip(head, 160)}")

# ── 5. Stima finale post-cleanup con cap 64k ─────────────────────────────────
keep = df[~dirty].copy()
keep["capped"] = keep["len"].clip(upper=64_000)
print("\n\n=== STIMA POST-CLEANUP ===")
print(f"  Record tenuti:        {len(keep):,}")
print(f"  Token senza cap:      ~{keep['len'].sum()/4/1e6:.1f}M")
print(f"  Token con cap 64k:    ~{keep['capped'].sum()/4/1e6:.1f}M")
print(f"  (rimossi {dirty.sum():,} sporchi, capati {(keep['len']>64000).sum():,} file giganti)")
