"""
SFT training su Modal A100-80GB — Qwen3.6-27B + DoRA r=128 (rsLoRA) bf16 + PACKING MANUALE.

Stessa logica di scripts/train_sft.py ma come Modal function (no quota, crediti free).
Cache pesi su Volume (no ri-download dei 54GB). Checkpoint locali + push finale + backup HF.

Usage:
  python -m modal run scripts/modal_train_sft.py                 # epoca piena
  python -m modal run scripts/modal_train_sft.py --max-steps 20  # smoke test (~10 min)
  python -m modal run scripts/modal_train_sft.py --load-4bit     # QLoRA (se OOM bf16)
"""

from __future__ import annotations

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("build-essential")   # gcc per i kernel Triton (bitsandbytes/optimizer)
    .pip_install(
        "torch==2.7.0",       # transformers 5.x richiede torch>=2.6 (float8_e8m0fnu)
        "transformers>=4.51", "peft>=0.13", "accelerate>=1.0",
        "bitsandbytes>=0.44", "datasets", "huggingface-hub", "sentencepiece", "protobuf",
    )
)
app = modal.App("qwen-cobol-sft-train", image=image)
model_vol = modal.Volume.from_name("qwen-cobol-model-cache", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3.6-27B"
SFT_REPO = "AlexThunder0/cobol-sft-dataset"
ADAPTER_REPO = "AlexThunder0/qwen-cobol-27b-sft"
SPLITS = ["mainframebench", "teacher_bulk", "alibaba_gold", "generate_spec_valid"]
MAX_LEN = 2048
HF_BACKUP_EVERY = 150
CACHE = "/models"


@app.function(
    gpu="A100-80GB",
    volumes={CACHE: model_vol},
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(max_steps: int = -1, load_4bit: bool = False) -> dict:
    import logging, os, glob
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset, concatenate_datasets, Dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    os.environ["HF_HOME"] = CACHE
    token = os.environ["HF_TOKEN"]

    def _repo_size_gb() -> float:
        from huggingface_hub import HfApi
        try:
            info = HfApi(token=token).repo_info(ADAPTER_REPO, repo_type="model", files_metadata=True)
            return sum((getattr(f, "size", 0) or 0) for f in info.siblings) / 1e9
        except Exception:
            return -1.0

    class HFBackup(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if model and state.global_step > 0 and state.global_step % HF_BACKUP_EVERY == 0:
                try:
                    model.push_to_hub(ADAPTER_REPO, token=token,
                                      commit_message=f"backup step {state.global_step}")
                    sz = _repo_size_gb()
                    logger.info("Backup HF @ step %d — repo adapter: %.2f GB", state.global_step, sz)
                except Exception as e:
                    # Push fallito (storage/rete) → NON ferma il training, salta il backup
                    logger.warning("Backup HF @ step %d FALLITO (training continua): %s",
                                   state.global_step, e)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token, cache_dir=CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    has_tmpl = getattr(tok, "chat_template", None) is not None

    # ── Carica e concatena split ──────────────────────────────────────────────
    dsets = []
    for s in SPLITS:
        try:
            d = load_dataset(SFT_REPO, split=s, token=token)
            dsets.append(d)
            logger.info("Split %s: %d", s, len(d))
        except Exception as e:
            logger.warning("Split %s skip: %s", s, e)
    dataset = concatenate_datasets(dsets)
    logger.info("Dataset totale: %d esempi", len(dataset))

    def render(msgs):
        if has_tmpl:
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass
        return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs)

    # ── Packing manuale ───────────────────────────────────────────────────────
    logger.info("Packing in blocchi da %d …", MAX_LEN)
    ids: list[int] = []
    for row in dataset:
        ids.extend(tok(render(row["messages"]), add_special_tokens=False)["input_ids"])
        ids.append(eos)
    nb = len(ids) // MAX_LEN
    packed = Dataset.from_dict({"input_ids": [ids[i*MAX_LEN:(i+1)*MAX_LEN] for i in range(nb)]})
    logger.info("Token %d → %d blocchi (1 epoca ≈ %d step)", len(ids), nb, nb // 16)

    # ── Modello + DoRA ────────────────────────────────────────────────────────
    lora = LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", use_dora=True, use_rslora=True,
    )
    # Ottimizzazioni ESATTE (zero impatto qualità): SDPA attention (kernel flash
    # integrati in PyTorch) + niente gradient checkpointing (a batch 2 su 80GB c'è
    # memoria → matematica identica, ~1.6x più veloce). Solo memoria/velocità.
    if load_4bit:
        logger.info("Carico 27B in 4-bit (QDoRA) + SDPA …")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                     device_map="auto", token=token,
                                                     torch_dtype=torch.bfloat16, cache_dir=CACHE,
                                                     attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        logger.info("Carico 27B in bf16 (DoRA) + SDPA, no gradient checkpointing …")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", token=token,
                                                     torch_dtype=torch.bfloat16, cache_dir=CACHE,
                                                     attn_implementation="sdpa")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir="/tmp/sft_out",
        num_train_epochs=1, max_steps=max_steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,  # eff 16, conservativo bf16
        learning_rate=2.0e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
        bf16=True, logging_steps=10, save_steps=50, save_total_limit=1,
        optim="paged_adamw_8bit", report_to="none", push_to_hub=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=packed,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                      callbacks=[HFBackup()])

    resume = bool(glob.glob("/tmp/sft_out/checkpoint-*"))
    logger.info("Training avviato (%d blocchi)%s …", nb, " — RESUME" if resume else "")
    trainer.train(resume_from_checkpoint=resume)

    logger.info("Push adapter finale → %s", ADAPTER_REPO)
    model.push_to_hub(ADAPTER_REPO, token=token)
    tok.push_to_hub(ADAPTER_REPO, token=token)
    logger.info("SFT completato.")
    return {"blocks": nb, "status": "done"}


@app.local_entrypoint()
def main(max_steps: int = -1, load_4bit: bool = False):
    print(f"Avvio SFT su Modal A100-80GB (max_steps={max_steps}, 4bit={load_4bit}) …")
    print(train.remote(max_steps=max_steps, load_4bit=load_4bit))
