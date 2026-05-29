"""
Supervised Fine-Tuning (SFT) with curriculum learning — W5.

Loads CPT adapter from HF Hub, applies DoRA r=64, trains on 40-60k ChatML examples.
Dataset is pre-sorted by difficulty_score (curriculum_sampler.build_curriculum).

Usage (Lightning L40S terminal):
    python -m src.train.sft --config config/training_qwen36_27b.yaml
"""

from __future__ import annotations

import argparse
import logging

import yaml
import wandb
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from src.synth.curriculum_sampler import build_curriculum

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sft(config_path: str) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    sft_cfg = cfg["sft"]
    ds_cfg = cfg["datasets"]

    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['wandb']['run_prefix']}-sft",
        config={**model_cfg, **sft_cfg},
    )

    # Load from CPT checkpoint
    cpt_hub_id = cfg["cpt"]["hub_model_id"]
    logger.info("Loading CPT checkpoint: %s", cpt_hub_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cpt_hub_id,
        max_seq_length=sft_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=sft_cfg["r"],
        lora_alpha=sft_cfg["lora_alpha"],
        lora_dropout=sft_cfg["lora_dropout"],
        target_modules=cfg["dora"]["target_modules"],
        use_dora=cfg["dora"]["use_dora"],
        bias=sft_cfg["bias"],
        use_gradient_checkpointing=sft_cfg["gradient_checkpointing"],
    )

    logger.info("Loading SFT dataset from %s", ds_cfg["sft_dataset"])
    # SFT dataset has splits: mainframebench, gemini_gold, bulk_teacher
    dataset = concatenate_datasets([
        load_dataset(ds_cfg["sft_dataset"], split=split)
        for split in ["mainframebench", "gemini_gold", "bulk_teacher"]
        if _split_exists(ds_cfg["sft_dataset"], split)
    ])

    if sft_cfg.get("curriculum"):
        logger.info("Applying curriculum ordering by difficulty_score …")
        dataset = build_curriculum(dataset)

    training_args = SFTConfig(
        output_dir="./outputs/sft",
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        weight_decay=sft_cfg["weight_decay"],
        optim=sft_cfg["optim"],
        bf16=sft_cfg["bf16"],
        fp16=sft_cfg["fp16"],
        logging_steps=sft_cfg["logging_steps"],
        save_steps=sft_cfg["save_steps"],
        save_total_limit=sft_cfg["save_total_limit"],
        push_to_hub=sft_cfg["push_to_hub"],
        hub_model_id=sft_cfg["hub_model_id"],
        report_to="wandb",
        max_seq_length=sft_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SFT training …")
    trainer.train()

    model.push_to_hub(sft_cfg["hub_model_id"])
    tokenizer.push_to_hub(sft_cfg["hub_model_id"])
    wandb.finish()


def _split_exists(repo: str, split: str) -> bool:
    try:
        load_dataset(repo, split=split, streaming=True)
        return True
    except Exception:
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SFT with curriculum")
    p.add_argument("--config", default="config/training_qwen36_27b.yaml")
    args = p.parse_args()
    run_sft(args.config)


if __name__ == "__main__":
    main()
