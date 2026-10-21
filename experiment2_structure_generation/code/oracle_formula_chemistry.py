#!/usr/bin/env python3

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Exact chemistry definitions copied from the current
# formula_candidate_pool_11elem_10ppm_fast_v3.py
# ============================================================

ATOMIC_MASS: Dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
    "Si": 27.97692653465,
}

PROTON_MASS = 1.007276466812

ELEMENT_PRINT_ORDER = (
    "C", "H", "N", "O", "P", "S",
    "F", "Cl", "Br", "I", "Si"
)


@dataclass(frozen=True)
class Adduct:
    name: str
    mass_shift: float
    priority: int
    prior_penalty: float


ADDUCTS: Dict[str, List[Adduct]] = {
    "positive": [
        Adduct("[M+H]+", PROTON_MASS, 1, 0.00),
        Adduct("[M+Na]+", 22.989218, 2, 0.30),
        Adduct("[M+NH4]+", 18.033823, 3, 0.45),
        Adduct("[M+K]+", 38.963158, 4, 0.55),
    ],
    "negative": [
        Adduct("[M-H]-", -PROTON_MASS, 1, 0.00),
        Adduct("[M+Cl]-", 34.969402, 2, 0.45),
        Adduct("[M+FA-H]-", 44.998201, 3, 0.35),
        Adduct("[M+Ac-H]-", 59.013851, 4, 0.40),
    ],
}


FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Optional[Dict[str, int]]:
    if not formula or not isinstance(formula, str):
        return None

    s = (
        formula.strip()
        .replace("+", "")
        .replace("-", "")
    )

    counts: Dict[str, int] = defaultdict(int)

    pos = 0

    for m in FORMULA_RE.finditer(s):

        if m.start() != pos:
            return None

        element, number = m.groups()

        if element not in ATOMIC_MASS:
            return None

        counts[element] += (
            int(number)
            if number
            else 1
        )

        pos = m.end()

    if pos != len(s) or not counts:
        return None

    return dict(counts)


def format_formula(counts: Dict[str, int]) -> str:
    out = []

    for element in ELEMENT_PRINT_ORDER:

        n = counts.get(element, 0)

        if n > 0:
            out.append(
                element
                if n == 1
                else f"{element}{n}"
            )

    return "".join(out)


def normalize_formula(formula: str) -> Optional[str]:
    counts = parse_formula(formula)

    if not counts:
        return None

    return format_formula(counts)


def exact_mass(counts: Dict[str, int]) -> float:
    return sum(
        ATOMIC_MASS[element] * n
        for element, n in counts.items()
    )


def normalize_polarity(value: Any) -> Optional[str]:
    if value is None:
        return None

    s = str(value).strip().lower()

    if s in {
        "positive",
        "pos",
        "+",
        "p",
    }:
        return "positive"

    if s in {
        "negative",
        "neg",
        "-",
        "n",
    }:
        return "negative"

    return None


def calc_dbe(counts: Dict[str, int]) -> float:
    c = counts.get("C", 0)
    si = counts.get("Si", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    p = counts.get("P", 0)

    halogens = sum(
        counts.get(element, 0)
        for element in (
            "F",
            "Cl",
            "Br",
            "I",
        )
    )

    return (
        1.0
        + c
        + si
        - h / 2.0
        + (n + p) / 2.0
        - halogens / 2.0
    )
