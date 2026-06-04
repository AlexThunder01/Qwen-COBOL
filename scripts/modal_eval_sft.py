"""
Modal eval — Qwen3.6-27B + SFT LoRA adapter su COBOLEval.

Differenze vs baseline (modal_eval.py):
  - Carica l'adapter LoRA `AlexThunder0/qwen-cobol-27b-sft` a runtime (vLLM enable_lora)
  - Usa il formato CHAT (il modello SFT è instruction-tuned, non base)
  - Estrae il COBOL dalla risposta dell'assistant (gestisce ```cobol fences)

Confronto target: baseline W1 (32.88% compile, 0% Pass@1 raw, 10.27% GOBACK-norm).

Usage:
  python -m modal run scripts/modal_eval_sft.py            # full 164
  python -m modal run scripts/modal_eval_sft.py --quick    # 10 problemi
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("git", "gnucobol")
    .pip_install("vllm>=0.8.0", "transformers>=4.50", "huggingface_hub>=0.26")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("qwen-cobol-sft-eval", image=image)
model_vol = modal.Volume.from_name("qwen-cobol-model-cache", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen3.6-27B"
SFT_ADAPTER = "AlexThunder0/qwen-cobol-27b-sft"
COBOLEVAL_REPO = "https://github.com/BloopAI/COBOLEval.git"
COBOLEVAL_DIR = "/coboleval"
COBOLEVAL_COMMIT = "0bb96c3114bb2bb28e221e9d6000614781f8609d"

SOTA_COMPILE = 0.7395
SOTA_PASS1 = 0.4933

# Due modalità per l'A/B test sul thinking:
INSTRUCTION_THINK = (
    "Complete the following COBOL program by implementing the PROCEDURE DIVISION.\n"
    "Reason step by step, then provide the complete, compilable COBOL program "
    "inside a single ```cobol code block at the end.\n\n"
    "```cobol\n{prompt}\n```"
)
INSTRUCTION_DIRECT = (
    "Complete the following COBOL program by implementing the PROCEDURE DIVISION. "
    "Provide the complete, compilable COBOL program inside a single ```cobol code block.\n\n"
    "```cobol\n{prompt}\n```"
)


def swap_sections(src: str) -> str:
    """begin → working_storage → linkage → procedure (identica a modal_eval.py)."""
    working_storage, linkage, procedure, begin = [], [], [], []
    current_section = begin
    for line in src.split("\n"):
        stripped = line.strip().upper()
        if stripped.startswith("WORKING-STORAGE SECTION."):
            current_section = working_storage
        elif stripped.startswith("LINKAGE SECTION."):
            current_section = linkage
        elif stripped.startswith("PROCEDURE DIVISION"):
            current_section = procedure
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        current_section.append(line)
    return "\n".join(begin + working_storage + linkage + procedure)


def extract_cobol(response: str, prompt: str) -> str:
    """
    Estrae il programma COBOL dalla risposta, robusto al thinking verboso.
    Priorità:
      1. blocco fenced (ultimo) che contiene IDENTIFICATION DIVISION → programma intero
      2. blocco fenced (ultimo) con PROCEDURE DIVISION → completion, concatena al prompt
      3. slice dal primo IDENTIFICATION DIVISION fino a fine (no fences)
      4. fallback: prompt + risposta intera
    """
    blocks = re.findall(r"```(?:cobol)?\s*\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)

    # 1. Programmi COMPLETI (IDENTIFICATION + PROCEDURE DIVISION) → prendi il più
    #    completo (più lungo). Robusto al rambling con blocchi parziali multipli.
    complete = [
        b.strip() for b in blocks
        if re.search(r"IDENTIFICATION\s+DIVISION", b, re.IGNORECASE)
        and re.search(r"PROCEDURE\s+DIVISION", b, re.IGNORECASE)
    ]
    if complete:
        return max(complete, key=len)

    # 2. Blocco con almeno IDENTIFICATION DIVISION
    for b in reversed(blocks):
        if re.search(r"IDENTIFICATION\s+DIVISION", b, re.IGNORECASE):
            return b.strip()

    # 3. Solo completion (PROCEDURE DIVISION) → concatena
    for b in reversed(blocks):
        if re.search(r"PROCEDURE\s+DIVISION", b, re.IGNORECASE):
            return prompt + "\n" + b.strip()

    # 4. Nessun fence: slice da IDENTIFICATION DIVISION fino al primo marker di turno
    m = re.search(r"(IDENTIFICATION\s+DIVISION.*)", response, re.DOTALL | re.IGNORECASE)
    if m:
        return re.split(r"<\|", m.group(1))[0].strip()

    return prompt + "\n" + response.strip()


@app.function(
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": model_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(n_problems: int | None = None, use_lora: bool = True, think: bool = False) -> dict:
    import os
    import shutil
    import sys

    from huggingface_hub import snapshot_download

    # ── Clone + patch COBOLEval (identico a baseline) ────────────────────────
    shutil.rmtree(COBOLEVAL_DIR, ignore_errors=True)
    subprocess.run(["git", "clone", "--quiet", COBOLEVAL_REPO, COBOLEVAL_DIR], check=True)
    subprocess.run(["git", "checkout", "--quiet", COBOLEVAL_COMMIT], cwd=COBOLEVAL_DIR, check=True)

    eval_py = Path(COBOLEVAL_DIR) / "scripts" / "evaluation.py"
    src = eval_py.read_text()
    patched = src.replace(
        '    # if not cmd(f"./call_{name}"):\n'
        '    #     logger.warning(f"Runtime error for {path}")\n'
        '    #     return False\n',
        '    if not cmd(f"./call_{name}"):\n'
        '        logger.warning(f"Runtime error for {path}")\n'
        '        return False\n',
    )
    assert patched != src, "Patch evaluation.py fallito"
    eval_py.write_text(patched)
    os.chdir(COBOLEVAL_DIR)

    jsonl_path = Path(COBOLEVAL_DIR) / "data" / "CobolEval.jsonl"
    problems = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if n_problems:
        problems = problems[:n_problems]
    print(f"Problemi caricati: {len(problems)}")

    sys.path.insert(0, f"{COBOLEVAL_DIR}/scripts")
    from evaluation import check_correctness  # noqa: E402

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    lora_req = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        print(f"Scarico adapter {SFT_ADAPTER} …")
        adapter_path = snapshot_download(repo_id=SFT_ADAPTER, cache_dir="/models/adapters")
        print(f"Adapter in {adapter_path}")
        lora_req = LoRARequest("cobol-sft", 1, adapter_path)

    mode = "base + LoRA SFT" if use_lora else "VANILLA instruct (no LoRA)"
    think_label = "THINKING ON" if think else "thinking off"
    print(f"Carico {BASE_MODEL} — {mode} — {think_label} …")
    # Thinking ON serve più contesto per finire il reasoning + risposta
    max_len = 14336 if think else 8192
    llm = LLM(
        model=BASE_MODEL,
        download_dir="/models",
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=0.92,
        enable_lora=use_lora,
        max_lora_rank=64,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Thinking ON: serve molto spazio per il reasoning + risposta. OFF: basta 2048.
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=8192 if think else 2048,
        stop=["<|user|>", "<|assistant|>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"],
    )

    instruction = INSTRUCTION_THINK if think else INSTRUCTION_DIRECT

    def build_chat(user_msg: str) -> str:
        # think=False → enable_thinking=False (se supportato). think=True → default ON.
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if not think:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}], enable_thinking=False, **kwargs)
            except TypeError:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg + " /no_think"}], **kwargs)
        return tokenizer.apply_chat_template([{"role": "user", "content": user_msg}], **kwargs)

    chat_prompts = [build_chat(instruction.format(prompt=p["prompt"])) for p in problems]

    print(f"Genero completions ({mode}) per {len(problems)} problemi …")
    gen_kwargs = {"lora_request": lora_req} if lora_req else {}
    outputs = llm.generate(chat_prompts, sampling, **gen_kwargs)
    responses = [o.outputs[0].text for o in outputs]

    # ── Valuta ───────────────────────────────────────────────────────────────
    results = []
    n_compile = 0
    for prob, response in zip(problems, responses):
        cobol = extract_cobol(response, prob["prompt"])
        full_program = swap_sections(cobol)

        compiles = passed = False
        try:
            r = check_correctness(prob, full_program, COBOLEVAL_DIR)
            compiled_list = r.get("compiled", [])
            compiles = bool(compiled_list) and all(compiled_list)
            passed = bool(r.get("all_passed", False))
        except Exception as e:
            print(f"  ⚠️  {prob['task_id']}: {type(e).__name__}: {e}")
        if compiles:
            n_compile += 1
        status = "PASS" if passed else ("COMPILE" if compiles else "FAIL")
        print(f"  {prob['task_id']:30s}  {status}")
        results.append({
            "task_id": prob["task_id"],
            "compiles": compiles,
            "passed": passed,
            "response": response,
        })

    n_pass = sum(1 for r in results if r["passed"])
    n = len(problems)
    label = "base + LoRA SFT" if use_lora else "VANILLA instruct chat"
    summary = {
        "model": f"{BASE_MODEL}" + (f" + {SFT_ADAPTER}" if use_lora else " (vanilla chat)"),
        "mode": label,
        "n_problems": n,
        "compile_rate": round(n_compile / n, 4),
        "pass_at_1": round(n_pass / n, 4),
        "n_compile": n_compile,
        "n_pass": n_pass,
        "w1_baseline_raw": {"compile_rate": 0.3288, "pass_at_1_raw": 0.0, "pass_at_1_goback": 0.1027},
        "sota": {"compile_rate": SOTA_COMPILE, "pass_at_1": SOTA_PASS1, "model": "COBOL-Coder"},
        "results": results,
    }

    print(f"\n{'='*54}")
    print(f"Modello:      {label}")
    print(f"Compile rate: {summary['compile_rate']*100:.1f}%  (SOTA 73.95%)")
    print(f"Pass@1:       {summary['pass_at_1']*100:.2f}%  (SOTA 49.33%)")
    print(f"{'='*54}")
    return summary


@app.local_entrypoint()
def main(quick: bool = False, n_problems: int = 0, vanilla: bool = False, think: bool = False):
    """--vanilla: instruct puro senza LoRA. --think: reasoning ON (max_tokens 8192)."""
    n = 10 if quick else (n_problems or None)
    result = run_eval.remote(n_problems=n, use_lora=not vanilla, think=think)
    name = "vanilla_chat" if vanilla else "sft_step300"
    name += "_think" if think else "_direct"
    out_path = Path(f"results/{name}_qwen36_27b.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRisultati salvati in {out_path}")
