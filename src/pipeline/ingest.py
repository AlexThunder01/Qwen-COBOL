"""
Data ingest pipeline — streams COBOL source files from public sources and
pushes validated parquet shards to HF Hub private dataset.

Ingest priority (W2):
  1. XMainframe Training Corpus (Fsoft-AIC/XMainframe, MIT)
  2. The Stack v2 dedup — filtered to COBOL
  3. X-COBOL Zenodo (182 repos, manually downloaded)
  4. Community repos (opensourcecobol, openmainframeproject, etc.)
  5. NIST COBOL 85 test suite (ships with gnucobol-bin)

Run on Kaggle CPU (32 cores). Outputs stream to HF Hub — never writes full
corpus to local disk simultaneously.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

from src.pipeline.clean import clean_record
from src.pipeline.dedupe import MinHashDeduper
from src.pipeline.difficulty_scorer import score_difficulty

logger = logging.getLogger(__name__)

HUB_CORPUS_REPO = "AlexThunder0/cobol-cpt-corpus"
SHARD_SIZE = 5_000


def iter_xmainframe() -> Iterator[dict]:
    """Stream XMainframe training corpus.

    Fsoft-AIC/XMainframe non è mai stato pubblicato come dataset HF nonostante
    il paper lo menzionasse. La funzione è un no-op — skip_xmainframe=True
    per default nel notebook Kaggle.
    """
    logger.warning("iter_xmainframe: dataset Fsoft-AIC/XMainframe non disponibile su HF Hub — skippato")
    return
    yield  # make it a generator


def iter_the_stack_v1() -> Iterator[dict]:
    """Stream COBOL da The Stack v1 (`bigcode/the-stack-dedup`) — content INLINE.

    Gated: accettare i termini su
    https://huggingface.co/datasets/bigcode/the-stack-dedup
    """
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/cobol",
        split="train",
        streaming=True,
    )
    for row in ds:
        content = row.get("content", "")
        if content:
            yield {
                "content": content,
                "source": "the-stack-v1",
                "path": row.get("max_stars_repo_path") or row.get("path", ""),
            }


def iter_the_stack_v2() -> Iterator[dict]:
    """Stream COBOL da The Stack v2 dedup (~36k file) scaricando il content da
    Software Heritage S3.

    The Stack v2 è metadata-only: ogni row ha `blob_id` + `src_encoding`, il
    codice va scaricato da `s3://softwareheritage/content/{blob_id}` (gzip).
    Serve un account AWS con credenziali in env:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    Volume atteso: ~36k file → ~40M token (10x la v1).
    """
    import boto3
    from smart_open import open as s3_open

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")

    def download_content(blob_id: str, src_encoding: str) -> str | None:
        url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with s3_open(url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                return fin.read().decode(src_encoding, errors="replace")
        except Exception as e:
            logger.debug("blob %s fallito: %s", blob_id, e)
            return None

    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        "COBOL",
        split="train",
        streaming=True,
    )
    n_ok = n_fail = 0
    for row in ds:
        blob_id = row.get("blob_id")
        if not blob_id:
            continue
        content = download_content(blob_id, row.get("src_encoding", "utf-8"))
        if content:
            n_ok += 1
            yield {
                "content": content,
                "source": "the-stack-v2",
                "path": row.get("path", ""),
            }
        else:
            n_fail += 1
        if (n_ok + n_fail) % 2000 == 0:
            logger.info("Stack v2: %d scaricati, %d falliti", n_ok, n_fail)


def iter_the_stack() -> Iterator[dict]:
    """Sceglie v2 (con download S3) se ci sono credenziali AWS, altrimenti v1."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.info("The Stack v2 (download S3) — credenziali AWS presenti")
        yield from iter_the_stack_v2()
    else:
        logger.info("The Stack v1 (content inline) — nessuna credenziale AWS")
        yield from iter_the_stack_v1()


def iter_xcobol_zenodo(zenodo_dir: Path) -> Iterator[dict]:
    """Yield COBOL files from the X-COBOL Zenodo archive (manually downloaded).
    Download from: https://zenodo.org/records/7968845
    """
    for ext in ("*.cob", "*.cbl", "*.cpy", "*.COB", "*.CBL", "*.CPY"):
        for f in zenodo_dir.rglob(ext):
            try:
                yield {"content": f.read_text(errors="replace"), "source": "x-cobol", "path": str(f)}
            except Exception:
                continue


def iter_community_repos(repos_dir: Path) -> Iterator[dict]:
    """Yield COBOL files from locally cloned community repos."""
    for ext in ("*.cob", "*.cbl", "*.cpy", "*.cblx", "*.COB", "*.CBL"):
        for f in repos_dir.rglob(ext):
            try:
                yield {"content": f.read_text(errors="replace"), "source": "community", "path": str(f)}
            except Exception:
                continue


def iter_nist(gnucobol_share_dir: Path) -> Iterator[dict]:
    """Yield NIST COBOL 85 test programs (public domain, ships with gnucobol-bin)."""
    nist_dir = gnucobol_share_dir / "testsuite"
    if not nist_dir.exists():
        logger.warning("NIST test suite not found at %s — skipping", nist_dir)
        return
    for f in nist_dir.rglob("*.cob"):
        try:
            yield {"content": f.read_text(errors="replace"), "source": "nist", "path": str(f)}
        except Exception:
            continue


def run_ingest(
    zenodo_dir: str = "/data/xcobol",
    repos_dir: str = "/data/community_repos",
    gnucobol_share: str = "/usr/share/gnucobol",
    skip_stack: bool = False,
    skip_xmainframe: bool = False,
) -> None:
    deduper = MinHashDeduper(threshold=0.7)
    all_records: list[dict] = []

    sources: list[Iterator[dict]] = []
    if not skip_xmainframe:
        sources.append(iter_xmainframe())
    if not skip_stack:
        sources.append(iter_the_stack())
    if Path(zenodo_dir).exists():
        sources.append(iter_xcobol_zenodo(Path(zenodo_dir)))
    if Path(repos_dir).exists():
        sources.append(iter_community_repos(Path(repos_dir)))
    sources.append(iter_nist(Path(gnucobol_share)))

    for source in sources:
        for raw in source:
            record = clean_record(raw)
            if record is None:
                continue
            if deduper.is_duplicate(record["content"]):
                continue
            deduper.add(record["content"])
            record["difficulty_score"] = score_difficulty(record["content"])
            all_records.append(record)
            if len(all_records) % SHARD_SIZE == 0:
                logger.info("Raccolti %d record finora…", len(all_records))

    logger.info("Ingest completo: %d record. Push su HF Hub…", len(all_records))
    ds = Dataset.from_list(all_records)
    ds.push_to_hub(HUB_CORPUS_REPO, split="train", private=True)
    logger.info("Push completato: %d record su %s", len(all_records), HUB_CORPUS_REPO)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Ingest COBOL corpus to HF Hub")
    p.add_argument("--zenodo-dir", default="/data/xcobol")
    p.add_argument("--repos-dir", default="/data/community_repos")
    p.add_argument("--gnucobol-share", default="/usr/share/gnucobol")
    p.add_argument("--skip-stack", action="store_true")
    p.add_argument("--skip-xmainframe", action="store_true")
    args = p.parse_args()
    run_ingest(args.zenodo_dir, args.repos_dir, args.gnucobol_share, args.skip_stack, args.skip_xmainframe)


if __name__ == "__main__":
    main()
