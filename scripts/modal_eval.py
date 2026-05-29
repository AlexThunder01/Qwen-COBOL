"""
Modal baseline eval — Qwen3.6-27B (base) su COBOLEval (164 problemi).

Misura il punto di partenza (vanilla, senza fine-tuning COBOL) da battere.
SOTA da battere: COBOL-Coder — 73.95% compile, 49.33% Pass@1.

Pre-requisiti:
  1. Crea il Modal secret con il tuo HF token (read-only):
       modal secret create huggingface-secret HF_TOKEN=hf_...
  2. Se il modello è gated, accetta i termini su huggingface.co prima di girare.

Usage:
  # Full eval — 164 problemi, ~30 min su A100-80GB (~$1.25):
  python -m modal run scripts/modal_eval.py

  # Smoke test — 10 problemi, ~5 min:
  python -m modal run scripts/modal_eval.py --quick

  # N problemi custom:
  python -m modal run scripts/modal_eval.py --n-problems 30

Output: results/baseline_qwen36_27b.json
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

# ── Image ──────────────────────────────────────────────────────────────────────
# Debian 13 (trixie) ha GnuCOBOL 4.x → supporta `-fformat=variable` come usato
# dal paper COBOLEval. Debian 12 ha solo 3.x e -free rompe i commenti fixed-format
# dei prompt (asterisco in col 7 + apostrofi in inglese → syntax error).
image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("git", "gnucobol")
    .pip_install(
        "vllm>=0.8.0",
        "transformers>=4.50",
        "huggingface_hub>=0.26",
    )
    # flashinfer compila kernel CUDA JIT → richiede nvcc che non è nell'image.
    # Disabilitato: vLLM usa il sampler PyTorch nativo, basta per inference.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("qwen-cobol-baseline-eval", image=image)

# Volume per cache pesi modello — evita ri-download ad ogni run
model_vol = modal.Volume.from_name("qwen-cobol-model-cache", create_if_missing=True)

# ── Costanti ───────────────────────────────────────────────────────────────────
# bf16 (54GB) → A100-80GB necessario. Nessuna quantizzazione = benchmark autorevole
# confrontabile con COBOL-Coder (misurato a piena precisione).
# Se Qwen/Qwen3.6-27B-Instruct diventa pubblico, sostituire MODEL_ID con quello.
MODEL_ID = "Qwen/Qwen3.6-27B"
COBOLEVAL_REPO = "https://github.com/BloopAI/COBOLEval.git"
COBOLEVAL_DIR = "/coboleval"
# Pin per riproducibilità — la patch .replace() su evaluation.py dipende dal
# testo esatto del file a questo commit.
COBOLEVAL_COMMIT = "0bb96c3114bb2bb28e221e9d6000614781f8609d"


# COBOL-Coder SOTA (paper arxiv 2604.03986)
SOTA_COMPILE = 0.7395
SOTA_PASS1 = 0.4933


def swap_sections(src: str) -> str:
    """
    Riordina le sezioni COBOL: begin → working_storage → linkage → procedure.
    Replica esattamente `scripts/generate.py::swap_sections` di COBOLEval.

    Necessario perché il prompt COBOLEval ha LINKAGE prima di WORKING-STORAGE
    (non-standard); senza questo riordino il programma non compila in cobc.
    Forza anche `PROCEDURE DIVISION USING LINKED-ITEMS.` come standardizzato
    dal paper.
    """
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


# ── Funzione principale (gira su Modal A100) ───────────────────────────────────
@app.function(
    gpu="A100-80GB",  # bf16 27B = 54GB → serve 80GB; ~$1.25 per full eval
    timeout=3600,
    volumes={"/models": model_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(n_problems: int | None = None) -> dict:
    import shutil
    import sys

    # ── Clone COBOLEval ─────────────────────────────────────────────────────
    # rmtree prima del clone: su container Modal warm /coboleval può già esistere
    # da una run precedente → git clone fallirebbe con "destination path exists".
    shutil.rmtree(COBOLEVAL_DIR, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--quiet", COBOLEVAL_REPO, COBOLEVAL_DIR],
        check=True,
    )
    # Pin del commit per riproducibilità: la patch .replace() su evaluation.py
    # dipende dal testo esatto del file. Fissando il commit evitiamo che un
    # cambiamento upstream rompa silenziosamente la pipeline.
    subprocess.run(
        ["git", "checkout", "--quiet", COBOLEVAL_COMMIT],
        cwd=COBOLEVAL_DIR, check=True,
    )
    print(f"COBOLEval clonato in {COBOLEVAL_DIR} @ {COBOLEVAL_COMMIT[:10]}")

    # ── Patch evaluation.py: abilita esecuzione binary ─────────────────────
    # Nel codice originale `exec()` esegue solo la compilazione, e il run del
    # binary è commentato per security ("untrusted model code"). Risultato: il
    # programma non gira mai, il file di output non viene scritto, Pass@1
    # rimane strutturalmente 0%. Siamo in container Modal isolato → safe.
    eval_py = Path(COBOLEVAL_DIR) / "scripts" / "evaluation.py"
    src = eval_py.read_text()
    patched_src = src.replace(
        '    # if not cmd(f"./call_{name}"):\n'
        '    #     logger.warning(f"Runtime error for {path}")\n'
        '    #     return False\n',
        '    if not cmd(f"./call_{name}"):\n'
        '        logger.warning(f"Runtime error for {path}")\n'
        '        return False\n',
    )
    assert patched_src != src, "Patch evaluation.py fallito: blocco commentato non trovato"
    eval_py.write_text(patched_src)
    print("Patchato evaluation.py: esecuzione binary abilitata")

    # `./call_<name>` è relativo → cwd deve essere /coboleval per trovare il
    # binary prodotto da cobc e leggere il file {NAME}.TXT scritto dal run.
    import os
    os.chdir(COBOLEVAL_DIR)
    print(f"CWD: {os.getcwd()}")

    # Sanity check: GnuCOBOL 3.2.0 supporta `-fformat=variable`.
    cobc_ver = subprocess.run(["cobc", "--version"], capture_output=True, text=True)
    print(f"GnuCOBOL: {cobc_ver.stdout.splitlines()[0] if cobc_ver.stdout else 'NOT FOUND'}")

    # ── Carica problemi da JSONL ─────────────────────────────────────────────
    # Struttura repo: data/CobolEval.jsonl — campi: task_id, prompt, entry_point,
    # canonical_solution, tests
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

    # ── Sanity check pipeline ────────────────────────────────────────────────
    # NB: il campo `canonical_solution` di COBOLEval contiene Python (ereditato
    # da HumanEval), non COBOL — quindi NON è usabile per validare il compile.
    # Validiamo invece che:
    #  (a) `cobc -fformat=variable` accetti un programma COBOL minimo valido
    #  (b) il prompt + una PROCEDURE DIVISION triviale compili → conferma che
    #      la struttura del prompt è ben formata e il nostro full_program è ok
    print("\n=== Esempio prompt (primo problema) ===")
    print(problems[0]["prompt"])
    print("=== /esempio ===\n")

    # Usiamo gli stessi flag del paper COBOLEval: `cobc -w -fformat=variable -x`.
    # Niente `-fsyntax-only` perché è più severo del compile reale (sembra che
    # enforci il section ordering anche quando il compile completo lo accetta).
    def try_compile(code: str, label: str) -> bool:
        with tempfile.NamedTemporaryFile(
            suffix=".cob", mode="w", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp_path = f.name
        cp = subprocess.run(
            ["cobc", "-w", "-fformat=variable", "-c", tmp_path],
            capture_output=True, text=True, timeout=15, cwd="/tmp",
        )
        # cleanup eventuali .o prodotti da -c
        Path(tmp_path).unlink(missing_ok=True)
        for ext in (".o", ".i"):
            Path(tmp_path).with_suffix(ext).unlink(missing_ok=True)
        ok = cp.returncode == 0
        print(f"  {'✅' if ok else '❌'} {label}")
        if not ok:
            print("    stderr:", cp.stderr.strip()[:400])
        return ok

    # (a) Cobol minimo hardcoded
    hello = (
        "       IDENTIFICATION DIVISION.\n"
        "       PROGRAM-ID. HELLO.\n"
        "       PROCEDURE DIVISION.\n"
        "           DISPLAY 'hi'.\n"
        "           STOP RUN.\n"
    )
    sanity_a = try_compile(hello, "hello-world minimo")

    # (b) prompt + stub PROCEDURE, passato attraverso swap_sections come fa
    # il paper. Se questo non compila, c'è ancora qualcosa di rotto nel setup.
    p0 = problems[0]
    stub_completion = (
        "       PROCEDURE DIVISION.\n"
        "           MOVE 0 TO RESULT.\n"
        "           GOBACK.\n"
    )
    swapped_stub = swap_sections(p0["prompt"] + stub_completion)
    sanity_b = try_compile(swapped_stub, f"{p0['task_id']} swap_sections + stub PROCEDURE")

    assert sanity_a, "cobc non compila nemmeno hello-world → setup compiler rotto"
    if not sanity_b:
        print("⚠️  Prompt + stub PROCEDURE non compila — il prompt potrebbe avere "
              "struttura non-banale (es. PROCEDURE DIVISION USING). Procedo comunque.")
    print()

    # ── Importa evaluator ufficiale ──────────────────────────────────────────
    sys.path.insert(0, f"{COBOLEVAL_DIR}/scripts")
    from evaluation import check_correctness  # noqa: E402

    # ── Carica modello ──────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams

    print(f"Carico {MODEL_ID} …")
    llm = LLM(
        model=MODEL_ID,
        download_dir="/models",
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=2048)

    # ── Genera completions ───────────────────────────────────────────────────
    # Modello base: input = p["prompt"], output = completion (parte mancante).
    # Programma completo = prompt + completion.
    prompts = [p["prompt"] for p in problems]
    print(f"Genero completions per {len(problems)} problemi …")
    outputs = llm.generate(prompts, sampling)
    completions = [o.outputs[0].text for o in outputs]

    # ── Valuta ──────────────────────────────────────────────────────────────
    results: list[dict] = []
    n_compile = 0

    for prob, completion in zip(problems, completions):
        # Programma completo = prompt + completion, poi riordinato con
        # swap_sections come fa scripts/generate.py del paper.
        # check_correctness scrive questo direttamente al .cbl senza modificare.
        full_program = swap_sections(prob["prompt"] + completion)

        # Eval ufficiale COBOLEval — ritorna dict con chiavi:
        #   compiled: list[bool]  un bool per test case (compila col caller)
        #   passed:   list[bool]  un bool per test case (test superato)
        #   all_passed: bool      tutti i passed True
        # Metrica autorevole: il programma compila se TUTTI i test compilano.
        compiles = False
        passed = False
        try:
            eval_result = check_correctness(prob, full_program, COBOLEVAL_DIR)
            compiled_list = eval_result.get("compiled", [])
            compiles = bool(compiled_list) and all(compiled_list)
            passed = bool(eval_result.get("all_passed", False))
        except Exception as e:
            print(f"  ⚠️  {prob['task_id']}: eval exception {type(e).__name__}: {e}")

        if compiles:
            n_compile += 1

        status = "PASS" if passed else ("COMPILE" if compiles else "FAIL")
        print(f"  {prob['task_id']:30s}  {status}")
        results.append({
            "task_id": prob["task_id"],
            "compiles": compiles,
            "passed": passed,
            "completion": completion,
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    n_pass = sum(1 for r in results if r["passed"])
    n = len(problems)
    summary = {
        "model": MODEL_ID,
        "n_problems": n,
        "compile_rate": round(n_compile / n, 4),
        "pass_at_1": round(n_pass / n, 4),
        "n_compile": n_compile,
        "n_pass": n_pass,
        "sota": {"compile_rate": SOTA_COMPILE, "pass_at_1": SOTA_PASS1, "model": "COBOL-Coder"},
        "results": results,
    }

    print(f"\n{'='*54}")
    print(f"Modello:      {MODEL_ID}")
    print(f"Problemi:     {n}")
    print(f"Compile rate: {summary['compile_rate']*100:.1f}%  (SOTA: {SOTA_COMPILE*100:.1f}%)")
    print(f"Pass@1:       {summary['pass_at_1']*100:.2f}%  (SOTA: {SOTA_PASS1*100:.2f}%)")
    print(f"{'='*54}")

    return summary


# ── Entrypoint locale ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(quick: bool = False, n_problems: int = 0):
    """
    --quick         smoke test con 10 problemi
    --n-problems N  numero custom di problemi (0 = tutti)
    """
    n = 10 if quick else (n_problems or None)
    result = run_eval.remote(n_problems=n)

    out_path = Path("results/baseline_qwen36_27b.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nRisultati salvati in {out_path}")
