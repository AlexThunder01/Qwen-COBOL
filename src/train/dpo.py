"""
Direct Preference Optimization (DPO) — W6.

Loads SFT adapter, trains on 3-5k compiler-validated preference pairs.
Key hyperparams: beta=0.1, LR=1e-5, 1-2 epochs.

Usage (Lightning L40S terminal):
    python -m src.train.dpo --config config/training_qwen36_27b.yaml
"""

from __future__ import annotations

import argparse
import logging

import yaml
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

PatchDPOTrainer()

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dpo(config_path: str) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    dpo_cfg = cfg["dpo"]
    ds_cfg = cfg["datasets"]

    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['wandb']['run_prefix']}-dpo",
        config={**model_cfg, **dpo_cfg},
    )

    sft_hub_id = cfg["sft"]["hub_model_id"]
    logger.info("Loading SFT checkpoint: %s", sft_hub_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_hub_id,
        max_seq_length=dpo_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=dpo_cfg["r"],
        lora_alpha=dpo_cfg["lora_alpha"],
        lora_dropout=dpo_cfg["lora_dropout"],
        target_modules=cfg["dora"]["target_modules"],
        use_dora=cfg["dora"]["use_dora"],
        bias=dpo_cfg["bias"],
        use_gradient_checkpointing=dpo_cfg["gradient_checkpointing"],
    )

    logger.info("Loading DPO dataset from %s", ds_cfg["dpo_dataset"])
    dataset = load_dataset(ds_cfg["dpo_dataset"], split="train")

    training_args = DPOConfig(
        output_dir="./outputs/dpo",
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        weight_decay=dpo_cfg["weight_decay"],
        optim=dpo_cfg["optim"],
        bf16=dpo_cfg["bf16"],
        fp16=dpo_cfg["fp16"],
        beta=dpo_cfg["beta"],
        logging_steps=dpo_cfg["logging_steps"],
        save_steps=dpo_cfg["save_steps"],
        save_total_limit=dpo_cfg["save_total_limit"],
        push_to_hub=dpo_cfg["push_to_hub"],
        hub_model_id=dpo_cfg["hub_model_id"],
        report_to="wandb",
        max_length=dpo_cfg["max_seq_length"],
        max_prompt_length=dpo_cfg["max_seq_length"] // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles reference model internally
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting DPO training …")
    trainer.train()

    model.push_to_hub(dpo_cfg["hub_model_id"])
    tokenizer.push_to_hub(dpo_cfg["hub_model_id"])
    wandb.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="DPO post-SFT")
    p.add_argument("--config", default="config/training_qwen36_27b.yaml")
    args = p.parse_args()
    run_dpo(args.config)


if __name__ == "__main__":
    main()
