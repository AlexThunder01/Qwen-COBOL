"""
Aggregate eval results and generate a leaderboard comparison table.
Prints a markdown table comparing Qwen-COBOL vs COBOL-Coder baseline.
"""

from __future__ import annotations

import json
from pathlib import Path


COBOL_CODER_BASELINE = {
    "model": "COBOL-Coder",
    "compile_rate": 0.7395,
    "pass_at_1": 0.4933,
    "javatrans_pass_at_1": 0.3493,
}

TARGET_STRETCH = {"compile_rate": 0.75, "pass_at_1": 0.50, "javatrans_pass_at_1": 0.36}


def generate_report(coboleval_path: str, javatrans_path: str | None = None) -> str:
    with open(coboleval_path) as f:
        coboleval = json.load(f)

    javatrans = {}
    if javatrans_path and Path(javatrans_path).exists():
        with open(javatrans_path) as f:
            javatrans = json.load(f)

    our_model = coboleval.get("model", "Qwen-COBOL")
    our_compile = coboleval.get("compile_rate", 0.0)
    our_pass1 = coboleval.get("pass_at_1", 0.0)
    our_java = javatrans.get("pass_at_1", None)

    lines = [
        "## Benchmark Results\n",
        "| Model | COBOLEval Compile | COBOLEval Pass@1 | Java→COBOL Pass@1 |",
        "|---|---|---|---|",
        f"| GPT-4o (reference) | 41.8% | 16.4% | ~0% |",
        f"| COBOL-Coder (SOTA) | {COBOL_CODER_BASELINE['compile_rate']*100:.1f}% | {COBOL_CODER_BASELINE['pass_at_1']*100:.1f}% | {COBOL_CODER_BASELINE['javatrans_pass_at_1']*100:.1f}% |",
        f"| **{our_model}** | **{our_compile*100:.1f}%** | **{our_pass1*100:.1f}%** | **{f'{our_java*100:.1f}%' if our_java is not None else 'N/A'}** |",
        f"| Target stretch | >{TARGET_STRETCH['compile_rate']*100:.0f}% | >{TARGET_STRETCH['pass_at_1']*100:.0f}% | >{TARGET_STRETCH['javatrans_pass_at_1']*100:.0f}% |",
    ]

    beats_compile = our_compile > COBOL_CODER_BASELINE["compile_rate"]
    beats_pass1 = our_pass1 > COBOL_CODER_BASELINE["pass_at_1"]
    lines.append(f"\n**Beats COBOL-Coder**: compile={'YES' if beats_compile else 'NO'}, pass@1={'YES' if beats_pass1 else 'NO'}")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--coboleval", required=True)
    p.add_argument("--javatrans", default=None)
    args = p.parse_args()
    print(generate_report(args.coboleval, args.javatrans))
