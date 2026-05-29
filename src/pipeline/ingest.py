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


def iter_the_stack_v2() -> Iterator[dict]:
    """Stream The Stack v2 dedup filtered to COBOL."""
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
    )
    for row in ds:
        lang = (row.get("programming_language") or "").lower()
        if lang == "cobol":
            yield {
                "content": row.get("content", ""),
                "source": "the-stack-v2",
                "path": row.get("path", ""),
            }


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


def _push_shard(api: HfApi, records: list[dict], idx: int) -> None:
    ds = Dataset.from_list(records)
    ds.push_to_hub(HUB_CORPUS_REPO, split=f"train", private=True, append=True)


def run_ingest(
    zenodo_dir: str = "/data/xcobol",
    repos_dir: str = "/data/community_repos",
    gnucobol_share: str = "/usr/share/gnucobol",
    skip_stack: bool = False,
    skip_xmainframe: bool = False,
) -> None:
    api = HfApi()
    deduper = MinHashDeduper(threshold=0.7)
    buffer: list[dict] = []
    total = 0

    sources: list[Iterator[dict]] = []
    if not skip_xmainframe:
        sources.append(iter_xmainframe())
    if not skip_stack:
        sources.append(iter_the_stack_v2())
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
            buffer.append(record)

            if len(buffer) >= SHARD_SIZE:
                _push_shard(api, buffer, total // SHARD_SIZE)
                total += len(buffer)
                logger.info("Pushed shard (%d records total)", total)
                buffer = []

    if buffer:
        _push_shard(api, buffer, total // SHARD_SIZE)
        total += len(buffer)

    logger.info("Ingest complete: %d records", total)


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
