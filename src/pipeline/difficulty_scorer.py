"""
Heuristic difficulty scorer for COBOL programs.
Used for curriculum learning: SFTTrainer receives dataset sorted by score ascending.

Score components (all normalized 0-1, weighted sum):
- LOC
- REDEFINES / OCCURS clause density
- CALL / PERFORM count
- GO TO count (spaghetti proxy)
- Cyclomatic complexity approximation (number of IF/EVALUATE/WHEN)
"""

from __future__ import annotations

import re


_LOC_WEIGHT = 0.15
_REDEFINES_WEIGHT = 0.20
_CALL_WEIGHT = 0.15
_GOTO_WEIGHT = 0.25
_BRANCH_WEIGHT = 0.25

# Caps for normalization
_LOC_CAP = 500
_REDEFINES_CAP = 20
_CALL_CAP = 30
_GOTO_CAP = 15
_BRANCH_CAP = 50


def score_difficulty(source: str) -> float:
    """Return a float in [0, 1] — higher means harder."""
    upper = source.upper()
    lines = source.splitlines()

    loc = min(len(lines), _LOC_CAP) / _LOC_CAP
    redefines = min(upper.count("REDEFINES"), _REDEFINES_CAP) / _REDEFINES_CAP
    calls = min(
        len(re.findall(r"\bCALL\b", upper)) + len(re.findall(r"\bPERFORM\b", upper)),
        _CALL_CAP,
    ) / _CALL_CAP
    gotos = min(len(re.findall(r"\bGO\s+TO\b", upper)), _GOTO_CAP) / _GOTO_CAP
    branches = min(
        len(re.findall(r"\bIF\b", upper))
        + len(re.findall(r"\bEVALUATE\b", upper))
        + len(re.findall(r"\bWHEN\b", upper)),
        _BRANCH_CAP,
    ) / _BRANCH_CAP

    return round(
        _LOC_WEIGHT * loc
        + _REDEFINES_WEIGHT * redefines
        + _CALL_WEIGHT * calls
        + _GOTO_WEIGHT * gotos
        + _BRANCH_WEIGHT * branches,
        4,
    )
