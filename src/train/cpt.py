"""
Continued Pre-Training (CPT) — W4.

Trains Qwen3.6-27B (or 14B fallback) on the COBOL corpus with DoRA r=128.
Dataset streams from HF Hub; no full corpus on local disk.

Usage (Lightning L40S terminal):
    # W1 sanity / throughput test (no HF corpus needed):
    python -m src.train.cpt --config config/training_qwen36_27b.yaml --dummy

    # Real CPT (corpus must exist on HF Hub):
    python -m src.train.cpt --config config/training_qwen36_27b.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml
import wandb
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

_DUMMY_SNIPPET = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SAMPLE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-COUNTER PIC 9(4) VALUE 0.
       01 WS-NAME    PIC X(30) VALUE SPACES.
       PROCEDURE DIVISION.
       MAIN-PARA.
           MOVE 'COBOL TRAINING' TO WS-NAME
           PERFORM VARYING WS-COUNTER FROM 1 BY 1
               UNTIL WS-COUNTER > 100
               DISPLAY WS-COUNTER ': ' WS-NAME
           END-PERFORM
           STOP RUN.
"""


def _make_dummy_dataset(n: int = 500) -> Dataset:
    return Dataset.from_dict({"text": [_DUMMY_SNIPPET] * n})


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cpt(config_path: str, dummy: bool = False) -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    cpt_cfg = cfg["cpt"]
    ds_cfg = cfg["datasets"]

    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['wandb']['run_prefix']}-cpt",
        config={**model_cfg, **cpt_cfg},
    )

    logger.info("Loading model: %s", model_cfg["name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=cpt_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cpt_cfg["r"],
        lora_alpha=cpt_cfg["lora_alpha"],
        lora_dropout=cpt_cfg["lora_dropout"],
        target_modules=cfg["dora"]["target_modules"],
        use_dora=cfg["dora"]["use_dora"],
        bias=cpt_cfg["bias"],
        use_gradient_checkpointing=cpt_cfg["gradient_checkpointing"],
    )

    if dummy:
        logger.info("Using dummy COBOL dataset for throughput test (500 samples)")
        dataset = _make_dummy_dataset(500)
        is_streaming = False
    else:
        logger.info("Streaming CPT corpus from %s", ds_cfg["cpt_corpus"])
        dataset = load_dataset(ds_cfg["cpt_corpus"], split="train", streaming=ds_cfg["streaming"])
        dataset = dataset.map(lambda row: {"text": row["content"]})
        is_streaming = ds_cfg["streaming"]

    # Streaming datasets have no __len__ — must use max_steps.
    # Dummy and non-streaming datasets can use num_train_epochs.
    extra_args: dict[str, Any] = {}
    if is_streaming:
        extra_args["max_steps"] = cpt_cfg.get("max_steps", 200)
    else:
        extra_args["num_train_epochs"] = cpt_cfg.get("num_train_epochs", 3)

    warmup_steps = int(extra_args.get("max_steps", 100) * cpt_cfg.get("warmup_ratio", 0.03))

    training_args = SFTConfig(
        output_dir="./outputs/cpt",
        per_device_train_batch_size=cpt_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cpt_cfg["gradient_accumulation_steps"],
        learning_rate=cpt_cfg["learning_rate"],
        lr_scheduler_type=cpt_cfg["lr_scheduler_type"],
        warmup_steps=max(warmup_steps, 1),
        weight_decay=cpt_cfg["weight_decay"],
        optim=cpt_cfg["optim"],
        bf16=cpt_cfg["bf16"],
        fp16=cpt_cfg["fp16"],
        logging_steps=cpt_cfg["logging_steps"],
        save_steps=cpt_cfg["save_steps"],
        save_total_limit=cpt_cfg["save_total_limit"],
        push_to_hub=cpt_cfg["push_to_hub"],
        hub_model_id=cpt_cfg["hub_model_id"],
        report_to="wandb",
        max_seq_length=cpt_cfg["max_seq_length"],
        dataset_text_field="text",
        **extra_args,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting CPT training …")
    trainer.train()

    logger.info("CPT complete. Pushing final adapter to HF Hub …")
    model.push_to_hub(cpt_cfg["hub_model_id"])
    tokenizer.push_to_hub(cpt_cfg["hub_model_id"])
    wandb.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Continued Pre-Training")
    p.add_argument("--config", default="config/training_qwen36_27b.yaml")
    p.add_argument("--dummy", action="store_true",
                   help="Use in-memory dummy COBOL data (W1 throughput test, no HF corpus needed)")
    args = p.parse_args()
    run_cpt(args.config, dummy=args.dummy)


if __name__ == "__main__":
    main()
