# GCP — Setup training SFT su A100

Pronto da eseguire appena la quota A100 è approvata. Regione: `us-central1`.

## Quota: punta alla A100-80GB
Per il training in **bf16** (qualità piena, più veloce) serve la **A100-80GB**:
quota `NVIDIA A100 80GB GPUs`. Se hai solo la A100-40GB (`NVIDIA A100 GPUs`),
usa il flag `--load-4bit` (QLoRA) — funziona, leggermente meno qualità/velocità.

## 1. Crea la VM (Deep Learning image, A100-80GB)

Dal Cloud Shell (icona `>_` in alto a destra nella console GCP):

```bash
gcloud compute instances create qwen-cobol-train \
  --zone=us-central1-a \
  --machine-type=a2-ultragpu-1g \
  --accelerator=type=nvidia-a100-80gb,count=1 \
  --image-family=common-cu124-debian-11 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=250GB \
  --metadata="install-nvidia-driver=True"
```

Note:
- `a2-ultragpu-1g` = 1× A100-**80GB**. (Per la 40GB: `a2-highgpu-1g` + `type=nvidia-tesla-a100`.)
- Immagine Deep Learning `common-cu124`: CUDA + driver + Python già pronti.
- Disco 250GB: i pesi 27B bf16 (~54GB) + cache.

## 2. SSH nella VM

```bash
gcloud compute ssh qwen-cobol-train --zone=us-central1-a
```

(Prima volta: installa il driver NVIDIA, attendi qualche minuto. Verifica: `nvidia-smi`.)

## 3. Setup repo + dipendenze

```bash
git clone https://github.com/AlexThunder01/Qwen-COBOL.git
cd Qwen-COBOL
pip install -q "torch" "transformers>=4.51" "peft>=0.13" datasets huggingface-hub \
    "accelerate>=1.0" "bitsandbytes>=0.44" sentencepiece protobuf
export HF_TOKEN=hf_...     # il tuo token HF
```

## 4. Smoke test (30 step, ~5 min)

```bash
python scripts/train_sft.py --max-steps 30
```

Controlla con `nvidia-smi` (altro terminale): bf16 27B ~ 60-70GB VRAM su 80GB.
Loss che scende. Se OOM, abbassa `per_device_train_batch_size` a 2 nello script.

## 5. Training pieno (1 epoca, bf16) — DENTRO tmux!

**Sempre in `tmux`**: così se cade la connessione SSH (Roma→Iowa, capita spesso),
il training continua sulla VM invece di morire.

```bash
tmux new -s train          # crea sessione tmux
export HF_TOKEN=hf_...      # (ri-settare dentro tmux)
python scripts/train_sft.py
# Stacca senza fermare: premi  Ctrl+B  poi  D
# Riattacca più tardi:  tmux attach -t train
```

- Checkpoint LOCALE ogni 50 step (`save_total_limit=1` → no saturazione disco).
- **Resume automatico**: se il processo crasha (VM viva), rilancia lo stesso comando →
  riprende dal checkpoint. NON ricomincia da zero.
- Adapter **finale** pushato su HF `AlexThunder0/qwen-cobol-27b-sft` (una volta sola,
  niente bloat git-LFS).
- Stima: ~400 step (packing) + DoRA su A100-80GB → ~1.5-2.5h.

### Safety stack (cosa copre cosa)
| Failure | Coperto da |
|---|---|
| Caduta SSH | **tmux** (il processo continua sulla VM) |
| Crash processo / OOM | checkpoint locale + **resume** (rilancia, riprende) |
| Stop manuale | resume |
| Morte VM (raro, non-spot) | nessun backup HF mid-run → re-run 2h (accettabile) |
| Deliverable | adapter finale su HF |

## 6. ⚠️ SPEGNI la VM appena finito (NON lasciarla accesa!)

```bash
gcloud compute instances stop qwen-cobol-train --zone=us-central1-a
# oppure elimina:
gcloud compute instances delete qwen-cobol-train --zone=us-central1-a
```

A100-80GB accesa ≈ €3-4/h. Spegnerla subito preserva i €258.

## 7. Eval (su Modal, harness già pronto)

Dal tuo PC dopo il training:
```bash
python -m modal run scripts/modal_eval_sft.py --think
```
Confronto col baseline vanilla: **20.55% Pass@1**.
