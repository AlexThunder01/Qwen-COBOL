# Prerequisites

All training, evaluation, and data-processing commands run inside **WSL2 Ubuntu 24.04**.
Windows-native Python is only for local debug notebooks.

## 1 — WSL2 Ubuntu 24.04

```powershell
# PowerShell (admin)
wsl --install -d Ubuntu-24.04
wsl --set-default Ubuntu-24.04
```

After first boot, inside WSL2:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git gnucobol docker.io docker-compose python3.11 python3.11-dev python3-pip

# Verify GnuCOBOL ≥ 3.2
cobc --version

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Docker in WSL2

```bash
sudo usermod -aG docker $USER
# restart shell
docker run hello-world
```

## 2 — tree-sitter-cobol

```bash
pip install tree-sitter
# tree-sitter-cobol binding
pip install tree-sitter-cobol
# Smoke test with 10 real COBOL files before W2
python -c "import tree_sitter_cobol; print('OK')"
```

If `tree-sitter-cobol` PyPI package doesn't exist yet, build from source:

```bash
git clone https://github.com/nickel-lang/tree-sitter-cobol   # adjust URL
cd tree-sitter-cobol && pip install .
```

## 3 — Accounts and API keys

| Service | URL | What you need |
|---|---|---|
| Lightning AI | https://lightning.ai | Free account; create L40S Studio; note monthly quota from dashboard |
| Kaggle | https://kaggle.com | Free account; enable GPU T4; download `kaggle.json` API key |
| Google AI Studio | https://aistudio.google.com | Free Gemini 2.5 Flash API key (1500 req/day) |
| HF Hub | https://huggingface.co | Free account; create private datasets `<user>/cobol-cpt-corpus`, `<user>/cobol-sft-dataset`, `<user>/cobol-dpo-dataset` |
| W&B | https://wandb.ai | Free account; run `wandb login` |

Copy `.env.example` → `.env` and fill in your keys.

## 4 — Python environment (WSL2)

```bash
cd ~/Qwen-COBOL   # or wherever you cloned on WSL2 side
uv sync
```

Key packages and minimum versions are in [pyproject.toml](pyproject.toml).

## 5 — GnuCOBOL smoke test

```bash
# Compile a hello-world COBOL program
cat > /tmp/hello.cob << 'EOF'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO.
       PROCEDURE DIVISION.
           DISPLAY "Hello, COBOL!".
           STOP RUN.
EOF
cobc -x /tmp/hello.cob -o /tmp/hello && /tmp/hello
# Expected: Hello, COBOL!
```

## 6 — Lightning Studio setup

1. Create new Studio → select L40S GPU
2. Open terminal in Studio:

```bash
pip install uv
git clone <your-hf-repo-or-github-url> Qwen-COBOL
cd Qwen-COBOL && uv sync
huggingface-cli login   # paste HF token
wandb login             # paste W&B key
```

3. **Storage note**: Lightning Drive is 10 GB.
   - Do NOT store raw corpus or base model weights in Drive
   - Base model downloads to `~/.cache/huggingface/` (outside Drive quota — verify in Studio)
   - Only DoRA adapter checkpoints (~500 MB–2 GB) live in Drive; push to HF Hub immediately after each epoch

## 7 — Decision gate (end of W1)

After sanity training (4–6h on L40S, 500 dummy samples):
- Measure token/sec
- Estimate total GPU-hours needed for CPT + SFT + DPO
- Compare with visible monthly quota in Lightning dashboard
- If 27B needs more hours than available → switch config to `training_fallback_14b.yaml` and proceed with Qwen3-14B
