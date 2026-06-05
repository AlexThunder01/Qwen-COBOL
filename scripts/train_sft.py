"""
SFT standalone — Qwen3.6-27B + DoRA (r=128, rsLoRA) + PACKING MANUALE.
Eseguibile su qualsiasi VM con GPU (GCP A100, ecc.). Niente Modal, niente unsloth.

Adapter: DoRA (qualità > LoRA, vicino al full-FT) + rsLoRA + rango 128 → qualità
massima robusta (abbiamo memoria/budget su A100-80GB). ~1.4x più lento di LoRA, ok.
Default: bf16 (A100-80GB, ~54GB, qualità piena, niente tassa dequant).
Fallback: --load-4bit (QDoRA, per A100-40GB).

Pipeline:
  1. Carica tutti gli split SFT (mainframebench + teacher_bulk + alibaba_gold + generate_spec_valid)
  2. Render ChatML → tokenizza → concatena con EOS → blocchi da 2048 (packing)
  3. LoRA + gradient checkpointing
  4. 1 epoca, push adapter su HF (checkpoint ogni 100 step + finale)

Env richieste: HF_TOKEN
Usage:
  python scripts/train_sft.py                  # bf16, epoca piena (A100-80GB)
  python scripts/train_sft.py --max-steps 30   # smoke test
  python scripts/train_sft.py --load-4bit      # QLoRA (A100-40GB)
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3.6-27B"
SFT_REPO = "AlexThunder0/cobol-sft-dataset"
ADAPTER_REPO = "AlexThunder0/qwen-cobol-27b-sft"
SPLITS = ["mainframebench", "teacher_bulk", "alibaba_gold", "generate_spec_valid"]
MAX_LEN = 2048
HF_BACKUP_EVERY = 150  # push adapter su HF ogni N step (backup anti morte-VM, bloat modesto)


class HFBackupCallback(TrainerCallback):
    """Push periodico dell'adapter su HF — backup contro la morte della VM.
    Pochi push (ogni HF_BACKUP_EVERY) → bloat git-LFS contenuto."""
    def __init__(self, repo: str, token: str, every: int):
        self.repo, self.token, self.every = repo, token, every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and state.global_step > 0 and state.global_step % self.every == 0:
            try:
                model.push_to_hub(self.repo, token=self.token,
                                  commit_message=f"backup step {state.global_step}")
                logger.info("Backup HF intermedio @ step %d", state.global_step)
            except Exception as e:
                logger.warning("Backup HF @ step %d fallito: %s", state.global_step, e)


def main(max_steps: int, load_4bit: bool) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN non settata")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    has_template = getattr(tokenizer, "chat_template", None) is not None

    # ── Carica split ──────────────────────────────────────────────────────────
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

    def render(msgs):
        if has_template:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass
        return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs)

    # ── Packing manuale ───────────────────────────────────────────────────────
    logger.info("Packing manuale in blocchi da %d …", MAX_LEN)
    all_ids: list[int] = []
    for row in dataset:
        all_ids.extend(tokenizer(render(row["messages"]), add_special_tokens=False)["input_ids"])
        all_ids.append(eos_id)
    n_blocks = len(all_ids) // MAX_LEN
    blocks = [all_ids[i * MAX_LEN:(i + 1) * MAX_LEN] for i in range(n_blocks)]
    packed = Dataset.from_dict({"input_ids": blocks})
    logger.info("Token: %d → %d blocchi (1 epoca = %d step con batch eff 16)",
                len(all_ids), len(blocks), len(blocks) // 16)

    # ── Modello: bf16 (default, A100-80GB) o 4-bit (fallback A100-40GB) ───────
    # DoRA (qualità > LoRA, vicino al full-FT) + rsLoRA (stabilizza rango alto) + r=128.
    # Abbiamo memoria/budget (A100-80GB, €258) → puntiamo alla qualità massima robusta.
    lora_cfg = LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        use_dora=True,
        use_rslora=True,
    )
    if load_4bit:
        logger.info("Carico %s in 4-bit (QLoRA) …", BASE_MODEL)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto",
            token=token, torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        logger.info("Carico %s in bf16 (LoRA piena precisione) …", BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto", token=token, torch_dtype=torch.bfloat16,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir="./sft_out",
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
        save_steps=50,          # checkpoint LOCALE ogni 50 step (per resume)
        save_total_limit=1,     # tiene solo l'ultimo → no saturazione disco VM
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=False,      # NO push checkpoint → evita bloat git-LFS su HF.
                                # L'adapter finale è pushato una volta sola sotto.
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=packed,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[HFBackupCallback(ADAPTER_REPO, token, HF_BACKUP_EVERY)],
    )
    # Resume automatico: se esiste un checkpoint in output_dir (es. dopo un crash
    # con la VM ancora viva), riprende da lì invece di ricominciare da zero.
    import glob
    ckpts = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    resume = bool(ckpts)
    logger.info("Training avviato (%d blocchi)%s …", len(packed),
                " — RESUME da checkpoint" if resume else "")
    trainer.train(resume_from_checkpoint=resume)

    logger.info("Push adapter finale → %s", ADAPTER_REPO)
    model.push_to_hub(ADAPTER_REPO, token=token)
    tokenizer.push_to_hub(ADAPTER_REPO, token=token)
    logger.info("SFT completato.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--load-4bit", action="store_true", help="QLoRA 4-bit (per A100-40GB); default bf16")
    a = p.parse_args()
    main(a.max_steps, a.load_4bit)
