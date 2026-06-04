"""
Supervised Fine-Tuning (SFT) con curriculum learning — W4.

Carica Qwen3.6-27B base direttamente (niente CPT checkpoint).
Dataset: AlexThunder0/cobol-sft-dataset (splits: mainframebench, teacher_bulk, alibaba_gold)
Metodo: DoRA r=64, QLoRA 4-bit, unsloth per velocità

Usage (Lightning terminal):
    pip install unsloth trl datasets wandb peft
    python -m src.train.sft --config config/training_qwen36_27b.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

import yaml
import wandb
from datasets import load_dataset, concatenate_datasets

from src.synth.curriculum_sampler import build_curriculum

logger = logging.getLogger(__name__)

SFT_SPLITS = ["mainframebench", "teacher_bulk", "alibaba_gold"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _split_exists(repo: str, split: str) -> bool:
    try:
        load_dataset(repo, split=split, streaming=True)
        return True
    except Exception:
        return False


def run_sft(config_path: str) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    sft_cfg = cfg["sft"]
    ds_cfg = cfg["datasets"]

    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['wandb']['run_prefix']}-sft",
        config={**model_cfg, **sft_cfg},
    )

    # Carica base model direttamente (niente CPT checkpoint)
    logger.info("Loading base model: %s", model_cfg["name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=sft_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    dora_cfg = cfg["dora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=sft_cfg["r"],
        lora_alpha=sft_cfg["lora_alpha"],
        lora_dropout=sft_cfg["lora_dropout"],
        target_modules=dora_cfg["target_modules"],
        use_dora=dora_cfg["use_dora"],
        bias=sft_cfg["bias"],
        use_gradient_checkpointing=sft_cfg["gradient_checkpointing"],
    )

    # Carica e concatena tutti gli split disponibili
    logger.info("Loading SFT dataset from %s …", ds_cfg["sft_dataset"])
    splits = [
        load_dataset(ds_cfg["sft_dataset"], split=split)
        for split in SFT_SPLITS
        if _split_exists(ds_cfg["sft_dataset"], split)
    ]
    dataset = concatenate_datasets(splits)
    logger.info("Dataset totale: %d esempi da %d splits", len(dataset), len(splits))

    if sft_cfg.get("curriculum"):
        logger.info("Curriculum ordering per difficulty_score …")
        dataset = build_curriculum(dataset)
    else:
        # SHUFFLE: senza curriculum, mischia per evitare bias da ordine
        # (concatenazione = mainframebench → teacher_bulk → alibaba_gold).
        # Cruciale se la run viene interrotta: il parziale resta rappresentativo.
        logger.info("Shuffle del dataset (no curriculum) …")
        dataset = dataset.shuffle(seed=42)

    # ── Rendering ChatML → colonna `text` ────────────────────────────────────
    # Usa il chat template del tokenizer se presente, altrimenti ChatML manuale
    # (il modello BASE potrebbe non avere un template).
    eos_token = tokenizer.eos_token or "<|im_end|>"
    has_template = getattr(tokenizer, "chat_template", None) is not None

    def render_chatml(msgs: list[dict]) -> str:
        if has_template:
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                pass
        # Fallback ChatML manuale
        parts = []
        for m in msgs:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        return "\n".join(parts) + eos_token

    logger.info("Rendering ChatML → text (template del tokenizer: %s) …", has_template)
    dataset = dataset.map(
        lambda ex: {"text": render_chatml(ex["messages"])},
        remove_columns=[c for c in dataset.column_names if c != "text"],
    )

    # warmup_steps invece di warmup_ratio (deprecato in TRL >= 0.10)
    total_steps = (
        len(dataset)
        // (sft_cfg["per_device_train_batch_size"] * sft_cfg["gradient_accumulation_steps"])
        * sft_cfg["num_train_epochs"]
    )
    warmup_steps = max(1, int(total_steps * sft_cfg["warmup_ratio"]))

    training_args = SFTConfig(
        output_dir="./outputs/sft",
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
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
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("SFT training avviato …")
    trainer.train()

    logger.info("Pushing adapter a HF Hub: %s", sft_cfg["hub_model_id"])
    model.push_to_hub(sft_cfg["hub_model_id"])
    tokenizer.push_to_hub(sft_cfg["hub_model_id"])
    wandb.finish()
    logger.info("SFT completato.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SFT con curriculum — W4")
    p.add_argument("--config", default="config/training_qwen36_27b.yaml")
    args = p.parse_args()
    run_sft(args.config)


if __name__ == "__main__":
    main()
