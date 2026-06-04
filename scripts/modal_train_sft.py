"""
W4 SFT training su Modal — Qwen3.6-27B + QLoRA, con PACKING MANUALE.

Perché Modal e non Lightning: budget Lightning esaurito, $12.70 su Modal.
Perché packing manuale: unsloth ignorava packing=True → 1116 step (troppo costoso).
Impacchettando i token in blocchi da 2048 a mano, 1 epoca ≈ ~400 step → ~$7-9 su H100.

Pipeline:
  1. Carica tutti gli split SFT (mainframebench + teacher_bulk + alibaba_gold + generate_spec)
  2. Render ChatML → testo
  3. Tokenizza tutto, concatena con EOS, chunka in blocchi da 2048 (packing)
  4. QLoRA 4-bit + gradient checkpointing (fit sicuro su 80GB)
  5. 1 epoca, push adapter su HF ogni 100 step (insurance) + alla fine

Usage:
  python -m modal run scripts/modal_train_sft.py
  python -m modal run scripts/modal_train_sft.py --max-steps 50   # smoke test
"""

from __future__ import annotations

import logging

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51.1",
        "peft>=0.13.0",
        "datasets",
        "huggingface-hub",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.44.0",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("qwen-cobol-sft", image=image)
model_vol = modal.Volume.from_name("qwen-cobol-model-cache", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3.6-27B"
SFT_REPO = "AlexThunder0/cobol-sft-dataset"
ADAPTER_REPO = "AlexThunder0/qwen-cobol-27b-sft"
SPLITS = ["mainframebench", "teacher_bulk", "alibaba_gold", "generate_spec"]
MAX_LEN = 2048


@app.function(
    gpu="H100",
    timeout=6 * 3600,
    volumes={"/models": model_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(max_steps: int = -1) -> dict:
    import os
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, concatenate_datasets, Dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    token = os.environ["HF_TOKEN"]

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token, cache_dir="/models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # ── Carica e concatena split ──────────────────────────────────────────────
    dsets = []
    for s in SPLITS:
        try:
            d = load_dataset(SFT_REPO, split=s, token=token)
            dsets.append(d)
            logger.info("Split %s: %d esempi", s, len(d))
        except Exception as e:
            logger.warning("Split %s non caricato: %s", s, e)
    dataset = concatenate_datasets(dsets)
    logger.info("Dataset totale: %d esempi", len(dataset))

    # ── Render ChatML → testo ─────────────────────────────────────────────────
    has_template = getattr(tokenizer, "chat_template", None) is not None

    def render(msgs):
        if has_template:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass
        return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs)

    # ── PACKING MANUALE: tokenizza tutto, concatena, chunka in blocchi da MAX_LEN
    logger.info("Packing manuale dei token in blocchi da %d …", MAX_LEN)
    all_ids: list[int] = []
    for row in dataset:
        text = render(row["messages"])
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(eos_id)

    n_blocks = len(all_ids) // MAX_LEN
    blocks = [all_ids[i * MAX_LEN:(i + 1) * MAX_LEN] for i in range(n_blocks)]
    packed = Dataset.from_dict({"input_ids": blocks})
    logger.info("Token totali: %d → %d blocchi da %d (1 epoca = %d blocchi)",
                len(all_ids), len(blocks), MAX_LEN, len(blocks))

    # ── Modello QLoRA 4-bit ───────────────────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Carico %s in 4-bit …", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        token=token,
        cache_dir="/models",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # ── Training ──────────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir="/models/sft_out",
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2.0e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=True,
        hub_model_id=ADAPTER_REPO,
        hub_strategy="checkpoint",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=packed,
        data_collator=collator,
    )
    logger.info("Training avviato (%d blocchi, batch eff 16) …", len(packed))
    trainer.train()

    logger.info("Push adapter finale → %s", ADAPTER_REPO)
    model.push_to_hub(ADAPTER_REPO, token=token)
    tokenizer.push_to_hub(ADAPTER_REPO, token=token)
    return {"blocks": len(packed), "status": "done"}


@app.local_entrypoint()
def main(max_steps: int = -1):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Avvio SFT su Modal H100 (QLoRA 4-bit + packing manuale) …")
    result = train.remote(max_steps=max_steps)
    print(f"Completato: {result}")
