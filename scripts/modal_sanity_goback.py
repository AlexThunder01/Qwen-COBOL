"""
Sanity check CPU-only: l'ipotesi STOP RUN vs GOBACK.

Riusa i completion già salvati (results/baseline_qwen36_27b.json) — NIENTE GPU,
niente rigenerazione. Per ogni completion testa due varianti:
  A) originale (così com'è uscito dal modello)
  B) con STOP RUN → GOBACK

Obiettivo diagnostico:
  - Se (B) fa passare qualche test che (A) non passava → conferma che lo 0%
    Pass@1 è (almeno in parte) causato dal bug STOP RUN nel sub-program, non
    da logica sbagliata. Conferma anche che il pipeline SA rilevare un PASS.
  - Se nemmeno (B) passa nulla → lo 0% è genuino (logica errata oltre STOP RUN).

Usage:
  python -m modal run scripts/modal_sanity_goback.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("git", "gnucobol")
    .pip_install("numpy", "loguru", "tqdm")  # dipendenze di COBOLEval scripts/evaluation.py
)
app = modal.App("qwen-cobol-sanity-goback", image=image)

COBOLEVAL_REPO = "https://github.com/BloopAI/COBOLEval.git"
COBOLEVAL_DIR = "/coboleval"


def swap_sections(src: str) -> str:
    """Identica a modal_eval.py: begin → working_storage → linkage → procedure."""
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


@app.function(timeout=900)
def run_sanity(completions_by_task: dict) -> dict:
    import os
    import sys

    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", COBOLEVAL_REPO, COBOLEVAL_DIR],
        check=True,
    )

    # Patch exec() per abilitare il run del binary (come in modal_eval.py)
    eval_py = Path(COBOLEVAL_DIR) / "scripts" / "evaluation.py"
    s = eval_py.read_text()
    s2 = s.replace(
        '    # if not cmd(f"./call_{name}"):\n'
        '    #     logger.warning(f"Runtime error for {path}")\n'
        '    #     return False\n',
        '    if not cmd(f"./call_{name}"):\n'
        '        logger.warning(f"Runtime error for {path}")\n'
        '        return False\n',
    )
    assert s2 != s, "patch exec() fallito"
    eval_py.write_text(s2)
    os.chdir(COBOLEVAL_DIR)

    jsonl_path = Path(COBOLEVAL_DIR) / "data" / "CobolEval.jsonl"
    problems = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                problems[p["task_id"]] = p

    sys.path.insert(0, f"{COBOLEVAL_DIR}/scripts")
    from evaluation import check_correctness  # noqa: E402

    def evaluate(prob, full_program):
        try:
            r = check_correctness(prob, full_program, COBOLEVAL_DIR)
            compiled = bool(r.get("compiled")) and all(r.get("compiled", []))
            passed = bool(r.get("all_passed", False))
            return compiled, passed
        except Exception as e:
            print(f"  exception: {type(e).__name__}: {e}")
            return False, False

    rows = []
    n_a_pass = n_b_pass = n_fixed = 0
    for task_id, completion in completions_by_task.items():
        prob = problems.get(task_id)
        if prob is None:
            continue
        base = prob["prompt"] + completion
        prog_a = swap_sections(base)
        prog_b = swap_sections(base.replace("STOP RUN", "GOBACK"))

        a_comp, a_pass = evaluate(prob, prog_a)
        b_comp, b_pass = evaluate(prob, prog_b)

        had_stop_run = "STOP RUN" in completion
        fixed = (not a_pass) and b_pass
        if a_pass:
            n_a_pass += 1
        if b_pass:
            n_b_pass += 1
        if fixed:
            n_fixed += 1

        flag = " <<< GOBACK FIX SBLOCCA PASS" if fixed else ""
        print(
            f"  {task_id:16s} STOP_RUN={int(had_stop_run)} "
            f"A(comp={int(a_comp)},pass={int(a_pass)}) "
            f"B(comp={int(b_comp)},pass={int(b_pass)}){flag}"
        )
        rows.append({
            "task_id": task_id,
            "had_stop_run": had_stop_run,
            "a_compiles": a_comp, "a_passed": a_pass,
            "b_compiles": b_comp, "b_passed": b_pass,
            "goback_fixed": fixed,
        })

    n = len(rows)
    print("\n" + "=" * 54)
    print(f"Completion testati:     {n}")
    print(f"PASS originali (A):     {n_a_pass}")
    print(f"PASS con GOBACK (B):    {n_b_pass}")
    print(f"Sbloccati da GOBACK:    {n_fixed}")
    print("=" * 54)
    return {"n": n, "a_pass": n_a_pass, "b_pass": n_b_pass, "fixed": n_fixed, "rows": rows}


@app.local_entrypoint()
def main():
    results_path = Path("results/baseline_qwen36_27b.json")
    data = json.loads(results_path.read_text())

    # Testiamo i compilanti (sono i candidati realistici a passare) + qualsiasi
    # completion contenente STOP RUN (potrebbe compilare solo dopo il fix).
    completions = {
        r["task_id"]: r["completion"]
        for r in data["results"]
        if r["compiles"] or "STOP RUN" in r["completion"]
    }
    print(f"Invio {len(completions)} completion al sanity check (CPU)…")
    out = run_sanity.remote(completions)

    out_path = Path("results/sanity_goback.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSalvato in {out_path}")
