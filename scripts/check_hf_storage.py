"""Riporta lo storage HF usato dai repo dell'utente (incl. history LFS)."""
import os, re
from pathlib import Path
raw = Path('.env').read_bytes().decode('latin-1')
os.environ['HF_TOKEN'] = re.search(r'hf_[A-Za-z0-9]{20,}', raw).group(0)
from huggingface_hub import HfApi

api = HfApi(token=os.environ['HF_TOKEN'])
me = api.whoami()
print(f"Account: {me.get('name')} | tipo: {me.get('type')}")

total = 0.0
print("\n=== Repo e storage usato (incl. versioni LFS in history) ===")
for kind in ("dataset", "model"):
    lister = api.list_datasets if kind == "dataset" else api.list_models
    for r in lister(author=me['name']):
        try:
            info = api.repo_info(r.id, repo_type=kind, files_metadata=True, token=os.environ['HF_TOKEN'])
            gb = sum((getattr(f, "size", 0) or 0) for f in info.siblings) / 1e9
            total += gb
            print(f"  [{kind:7s}] {r.id:45s} {gb*1000:8.1f} MB")
        except Exception as e:
            print(f"  [{kind:7s}] {r.id:45s} (errore: {str(e)[:40]})")

print(f"\n>>> TOTALE storage usato: {total:.2f} GB")
print(">>> Limite free tier: controlla https://huggingface.co/settings/storage")
