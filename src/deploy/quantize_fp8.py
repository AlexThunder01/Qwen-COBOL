"""
FP8 weight+activation quantization via llmcompressor → vLLM-compatible output (~27GB).
Target: L40S / H100 deployment (higher quality than AWQ).

Usage (Lightning L40S):
    python -m src.deploy.quantize_fp8 \
        --merged-dir ./outputs/merged \
        --output-dir ./outputs/fp8 \
        --push-to-hub YOUR_HF_USERNAME/qwen-cobol-27b-fp8
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def quantize_fp8(merged_dir: str, output_dir: str, push_to_hub: str | None = None) -> None:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    logger.info("Loading merged model from %s", merged_dir)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8",        # FP8 weight+activation — needs calibration data
        ignore=["lm_head"],
    )

    # TODO W7: provide ~512 COBOL calibration samples for activation quantization
    oneshot(
        model=merged_dir,
        recipe=recipe,
        output_dir=output_dir,
        # dataset=calibration_dataset,
        # num_calibration_samples=512,
    )

    if push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(folder_path=output_dir, repo_id=push_to_hub, private=True)
        logger.info("Pushed FP8 model to %s", push_to_hub)

    logger.info("FP8 quantization complete → %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="FP8 weight+activation quantization")
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--output-dir", default="./outputs/fp8")
    p.add_argument("--push-to-hub", default=None)
    args = p.parse_args()
    quantize_fp8(args.merged_dir, args.output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
