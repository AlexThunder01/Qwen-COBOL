"""
AWQ 4-bit quantization via llmcompressor → vLLM-compatible output (~15GB).
Target: consumer GPU deployment.

llmcompressor >= 0.10 handles AWQ natively (derived from AutoAWQ).

Usage (Lightning L40S):
    python -m src.deploy.quantize_awq \
        --merged-dir ./outputs/merged \
        --output-dir ./outputs/awq \
        --push-to-hub YOUR_HF_USERNAME/qwen-cobol-27b-awq
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_awq(merged_dir: str, output_dir: str, push_to_hub: str | None = None) -> None:
    from llmcompressor import oneshot  # uv sync --extra quant
    from llmcompressor.modifiers.quantization import QuantizationModifier

    logger.info("Loading merged model from %s", merged_dir)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",      # AWQ: 4-bit weights, 16-bit activations
        ignore=["lm_head"],
    )

    oneshot(
        model=merged_dir,
        recipe=recipe,
        output_dir=output_dir,
    )

    if push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(folder_path=output_dir, repo_id=push_to_hub, private=True)
        logger.info("Pushed AWQ model to %s", push_to_hub)

    logger.info("AWQ quantization complete → %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="AWQ 4-bit quantization")
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--output-dir", default="./outputs/awq")
    p.add_argument("--push-to-hub", default=None)
    args = p.parse_args()
    quantize_awq(args.merged_dir, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
