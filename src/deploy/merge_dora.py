"""
Merge CPT + SFT + DPO DoRA adapters into the base model for deployment.

Usage (Lightning L40S):
    python -m src.deploy.merge_dora \
        --base Qwen/Qwen3.6-27B-Instruct \
        --adapter YOUR_HF_USERNAME/qwen-cobol-27b-dpo \
        --output ./outputs/merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def merge_and_save(base_model: str, adapter_id: str, output_dir: str) -> None:
    logger.info("Loading base + adapter: %s + %s", base_model, adapter_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_id,  # Unsloth loads base + adapter together
        max_seq_length=8192,
        load_in_4bit=False,  # merge requires full precision
        dtype=None,
    )

    logger.info("Merging adapters into base weights …")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s", out)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Merge complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Merge DoRA adapters into base model")
    p.add_argument("--base", default="Qwen/Qwen3.6-27B-Instruct")
    p.add_argument("--adapter", required=True, help="HF Hub adapter ID (DPO checkpoint)")
    p.add_argument("--output", default="./outputs/merged")
    args = p.parse_args()
    merge_and_save(args.base, args.adapter, args.output)


if __name__ == "__main__":
    main()
