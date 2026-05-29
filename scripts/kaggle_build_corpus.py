"""
Kaggle CPU script — W2 corpus build (v2: git-clone based).

Strategia:
  1. Carica metadata da The Stack v2 COBOL parquet → estrae lista repo unici
  2. Git clone --depth 1 in parallelo di ogni repo
  3. Walk del repo, estrae .cbl/.cob/.cpy (inclusi copybook)
  4. Clean, dedup MinHash, push parquet su HF Hub

Vantaggi vs SWH API:
  - ~50x più veloce (1 clone vs N richieste SWH)
  - Niente rate limit
  - Include copybook insieme ai programmi principali
  - Più file per unit-of-work

Setup:
  1. Aggiungi HF_TOKEN come Kaggle Secret
  2. Acceleratore: None (CPU)
  3. Incolla in una cella ed esegui
"""

# ── Install ────────────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "datasets>=3.0", "huggingface-hub>=0.26", "datasketch",
    "pyarrow>=15.0", "tqdm",
], check=True)

# ── Imports ────────────────────────────────────────────────────────────────────
import re, logging, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
HUB_REPO = "AlexThunder0/cobol-cpt-corpus"
TMP_DIR = Path("/tmp/cobol_repos")
LOCAL_PARQUET = Path("/kaggle/working/corpus.parquet")
MAX_WORKERS = 8
CLONE_TIMEOUT = 90        # sec per repo
MAX_FILES_PER_REPO = 500  # protezione contro repo anomali
MIN_BYTES = 50
MAX_BYTES = 16_384 * 4
DEDUP_THRESHOLD = 0.7

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=HUB_REPO, repo_type="dataset", private=True, exist_ok=True)
logger.info("Dataset repo ready: %s", HUB_REPO)

TMP_DIR.mkdir(parents=True, exist_ok=True)
COBOL_EXTS = ("*.cob", "*.cbl", "*.cpy", "*.cblx",
              "*.COB", "*.CBL", "*.CPY", "*.CBLX")

# ── Step 1: estrai lista repo unici da The Stack v2 ───────────────────────────
logger.info("Carico metadata The Stack v2 COBOL …")
ds = load_dataset(
    "bigcode/the-stack-v2-dedup",
    data_dir="data/COBOL",
    split="train",
    streaming=True,
)

repos: set[str] = set()
total_files_metadata = 0
for row in ds:
    repo_name = row.get("repo_name", "")
    if repo_name:
        repos.add(repo_name)
        total_files_metadata += 1

logger.info("Metadata: %d file COBOL su %d repo unici", total_files_metadata, len(repos))

# ── Step 2: clone parallelo + estrazione ──────────────────────────────────────
def clone_and_extract(repo_name: str) -> list[dict]:
    """Clone --depth 1, estrae file COBOL, elimina repo."""
    safe_name = repo_name.replace("/", "__")
    repo_dir = TMP_DIR / safe_name

    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet",
             f"https://github.com/{repo_name}.git", str(repo_dir)],
            check=True, capture_output=True, timeout=CLONE_TIMEOUT,
        )
    except Exception:
        return []

    files: list[dict] = []
    try:
        for ext in COBOL_EXTS:
            for f in repo_dir.rglob(ext):
                if len(files) >= MAX_FILES_PER_REPO:
                    break
                try:
                    content = f.read_text(errors="replace")
                    if len(content.encode()) > MAX_BYTES * 2:
                        continue
                    files.append({
                        "content": content,
                        "source": "github-clone",
                        "path": str(f.relative_to(repo_dir)),
                        "repo_name": repo_name,
                    })
                except Exception:
                    continue
            if len(files) >= MAX_FILES_PER_REPO:
                break
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)

    return files


all_records: list[dict] = []
clone_ok = 0
clone_fail = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(clone_and_extract, r): r for r in repos}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="cloning"):
        records = fut.result()
        if records:
            clone_ok += 1
            all_records.extend(records)
        else:
            clone_fail += 1

logger.info("Clone: %d ok, %d falliti. File raccolti: %d", clone_ok, clone_fail, len(all_records))

# ── Step 3: clean + dedup ─────────────────────────────────────────────────────
_COMMENT_INDICATOR = frozenset({"*", "/"})

def clean(content: str) -> str | None:
    if not isinstance(content, str):
        return None
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        if len(line) >= 7:
            indicator = line[6]
            body = line[7:72].rstrip() if len(line) > 7 else ""
            if indicator in _COMMENT_INDICATOR:
                cleaned.append(f"      {indicator} {body}")
            else:
                cleaned.append(f"       {body}")
        else:
            cleaned.append(line)
    out, blanks = [], 0
    for line in cleaned:
        if line.strip() == "":
            blanks += 1
            if blanks <= 2:
                out.append("")
        else:
            blanks = 0
            out.append(line.rstrip())
    result = "\n".join(out).strip()
    b = len(result.encode())
    if b < MIN_BYTES or b > MAX_BYTES:
        return None
    return result


_TOK_RE = re.compile(r"[A-Z0-9\-_]+")

class Deduper:
    def __init__(self, threshold=0.7, num_perm=128):
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._n = 0
        self._num_perm = num_perm
    def _mh(self, text):
        m = MinHash(num_perm=self._num_perm)
        for tok in _TOK_RE.findall(text.upper()):
            m.update(tok.encode())
        return m
    def is_dup(self, text): return bool(self._lsh.query(self._mh(text)))
    def add(self, text):
        self._lsh.insert(str(self._n), self._mh(text))
        self._n += 1


def score_difficulty(content: str) -> float:
    u = content.upper()
    return round(
        min(content.count("\n") / 500, 1.0) * 0.3
        + min(u.count("REDEFINES") / 10, 1.0) * 0.2
        + min(u.count("CALL ") / 5, 1.0) * 0.2
        + min(u.count("GO TO") / 5, 1.0) * 0.15
        + min((u.count(" IF ") + u.count(" WHEN ") + u.count(" EVALUATE ")) / 20, 1.0) * 0.15,
        4,
    )


deduper = Deduper(threshold=DEDUP_THRESHOLD)
final_records: list[dict] = []
n_filtered = n_dup = 0

for raw in tqdm(all_records, desc="cleaning"):
    content = clean(raw["content"])
    if content is None:
        n_filtered += 1
        continue
    if deduper.is_dup(content):
        n_dup += 1
        continue
    deduper.add(content)
    final_records.append({
        "content": content,
        "source": raw["source"],
        "path": raw["path"],
        "repo_name": raw["repo_name"],
        "difficulty_score": score_difficulty(content),
    })

logger.info("Finali: %d (filtrati per size: %d, duplicati: %d)",
            len(final_records), n_filtered, n_dup)

# ── Step 4: push singolo su HF Hub ────────────────────────────────────────────
if final_records:
    logger.info("Pushing %d record su %s …", len(final_records), HUB_REPO)
    Dataset.from_list(final_records).push_to_hub(HUB_REPO, split="train", private=True)
    logger.info("Push completato.")

    # Stima token (~4 char per token su COBOL)
    total_chars = sum(len(r["content"]) for r in final_records)
    est_tokens = total_chars / 4
    logger.info("Token stimati: %.1fM (%.0f MB di testo)", est_tokens / 1e6, total_chars / 1e6)

print(f"\nCorpus su: https://huggingface.co/datasets/{HUB_REPO}")
print(f"Record: {len(final_records)}")
