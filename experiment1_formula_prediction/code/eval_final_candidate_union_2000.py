#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the proposed FINAL pre-LLM molecular-formula candidate pool.

FINAL scheme
============
Run three complementary branches for EVERY spectrum:

1) V3 main branch
   - 10 ppm
   - 11 neutral-formula elements:
     C,H,N,O,P,S,F,Cl,Br,I,Si
   - improved carbon-range heuristic
   - train-derived F/I/Si extension signatures
   - take Top-10

2) OLD complementary branch
   - 5 ppm
   - C,H,N,O,P,S,Cl,Br
   - original practical 3-branch candidate pool
   - take Top-10

3) Direct fragment-aware branch
   - directly imports the user's current rule file
   - 10 ppm
   - C,H,N,O
   - uses the rule file's fragment-formula explanation / ranking
   - take Top-5

FINAL candidate set:
    V3 Top-10 ∪ OLD Top-10 ∪ DIRECT Top-5
    -> deduplicate by neutral molecular formula
    -> NO further deterministic truncation
    -> at most 25 formulas
    -> this set is intended to be passed to the retrained LLM

Why no extra RRF/Top-20 compression?
------------------------------------
The purpose of the pre-LLM stage is candidate coverage, not final ranking.
Since 10+10+5 <= 25 and formulas are short, retaining the full deduplicated union
avoids throwing away a correct formula before the LLM sees it.

This evaluator reports:
- V3 Top-10 candidate coverage
- OLD Top-10 candidate coverage
- DIRECT Top-5 candidate coverage
- V3 ∪ OLD coverage
- V3 ∪ OLD ∪ DIRECT coverage  <-- proposed FINAL
- marginal rescued samples contributed by OLD and DIRECT
- average / median / P90 / P95 / max union size
- F/I/Si subset coverage

Leakage policy
--------------
- Exact annotated adduct is never supplied to candidate generation.
- If explicit polarity is unavailable, only the adduct SIGN is used.
- Reference molecular formula is used only for offline evaluation.
- Train formulas are used only to build one GLOBAL set of F/I/Si element-type
  signatures. No test formula is used for sample-level routing or template choice.
- LLM and SIRIUS rescue are both OFF.

Python 3.13
-----------
Dynamic import registers the rule module in sys.modules before exec_module(),
avoiding the dataclass import error seen previously.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import re
import signal
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Constants
# =============================================================================

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

NOMINAL_MASS: Dict[str, int] = {
    "H": 1, "C": 12, "N": 14, "O": 16, "P": 31, "S": 32,
    "F": 19, "Cl": 35, "Br": 79, "I": 127, "Si": 28,
}

PROTON_MASS = 1.007276466812

ELEMENT_PRINT_ORDER = ("C","H","N","O","P","S","F","Cl","Br","I","Si")
FINAL_RARE = ("P","S","F","Cl","Br","I","Si")
NEW_ELEMENTS = ("F","I","Si")

FINAL_ELEMENTS = ("C","H","N","O","P","S","F","Cl","Br","I","Si")

DEFAULT_LIMITS = {
    "C": 80,
    "H": 160,
    "N": 20,
    "O": 30,
    "P": 6,
    "S": 8,
    "F": 30,
    "Cl": 6,
    "Br": 4,
    "I": 3,
    "Si": 4,
}


@dataclass(frozen=True)
class Adduct:
    name: str
    mass_shift: float
    priority: int
    prior_penalty: float


ADDUCTS = {
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


@dataclass
class Candidate:
    formula: str
    adduct: str
    ppm_error: float
    dbe: float
    counts: Dict[str, int]
    element_level: int
    rule_valid: bool
    nitrogen_rule_valid: bool
    ratio_valid: bool
    isotope_prior_penalty: float
    adduct_prior_penalty: float


FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


# =============================================================================
# Dataset helpers
# =============================================================================

def get_nested(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_formula(formula):
    if not formula or not isinstance(formula, str):
        return None
    s = formula.strip().replace("+", "").replace("-", "")
    counts = defaultdict(int)
    pos = 0
    for m in FORMULA_RE.finditer(s):
        if m.start() != pos:
            return None
        e, n = m.groups()
        if e not in ATOMIC_MASS:
            return None
        counts[e] += int(n) if n else 1
        pos = m.end()
    if pos != len(s) or not counts:
        return None
    return dict(counts)


def format_formula(counts):
    out = []
    for e in ELEMENT_PRINT_ORDER:
        n = counts.get(e, 0)
        if n > 0:
            out.append(e if n == 1 else f"{e}{n}")
    return "".join(out)


def normalize_formula(formula):
    c = parse_formula(formula)
    return format_formula(c) if c else None


def formula_key(formula_or_counts):
    if isinstance(formula_or_counts, dict):
        c = formula_or_counts
    else:
        c = parse_formula(str(formula_or_counts))
    if not c:
        return None
    return tuple(sorted((e, int(n)) for e, n in c.items() if int(n) > 0))


def exact_mass(c):
    return sum(ATOMIC_MASS[e] * n for e, n in c.items())


def nominal_mass(c):
    return sum(NOMINAL_MASS[e] * n for e, n in c.items())


def normalize_polarity(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"positive","pos","+","p"}:
        return "positive"
    if s in {"negative","neg","-","n"}:
        return "negative"
    return None


def polarity_from_adduct(x):
    if x is None:
        return None
    s = str(x).strip()
    if s.endswith("+"):
        return "positive"
    if s.endswith("-"):
        return "negative"
    return None


def get_true_formula(item):
    vals = [
        get_nested(item, ["molecule","formula"]),
        item.get("true_formula"),
        item.get("formula"),
        get_nested(item, ["metadata","formula"]),
    ]
    for x in vals:
        if x:
            f = normalize_formula(str(x))
            if f:
                return f
    return None


def get_precursor_mz(item):
    vals = [
        get_nested(item, ["metadata","precursor_mz"]),
        item.get("precursor_mz"),
        get_nested(item, ["spectrum","precursor_mz"]),
    ]
    for x in vals:
        try:
            if x is not None:
                return float(x)
        except Exception:
            pass
    return None


def get_polarity(item):
    vals = [
        get_nested(item, ["metadata","ion_mode"]),
        get_nested(item, ["metadata","polarity"]),
        item.get("ion_mode"),
        item.get("polarity"),
    ]
    for x in vals:
        p = normalize_polarity(x)
        if p:
            return p

    # sign-only fallback
    vals = [
        get_nested(item, ["metadata","adduct"]),
        item.get("adduct"),
        item.get("true_adduct"),
    ]
    for x in vals:
        p = polarity_from_adduct(x)
        if p:
            return p
    return None


def get_fold(item):
    return item.get("fold") or get_nested(item, ["metadata","fold"])


def get_identifier(item, idx):
    return str(item.get("identifier") or item.get("id") or f"record_{idx}")


def get_fragments_and_intensities(item):
    mzs = get_nested(item, ["spectrum","mzs"])
    ints = get_nested(item, ["spectrum","intensities"])

    if mzs is None:
        mzs = item.get("ms2_mzs") or item.get("fragments") or []
    if ints is None:
        ints = item.get("ms2_intensities") or []

    peaks = get_nested(item, ["spectrum","peaks"])
    if not mzs and isinstance(peaks, list):
        valid = [p for p in peaks if isinstance(p,dict) and p.get("mz") is not None]
        mzs = [p["mz"] for p in valid]
        ints = [p.get("intensity", 1.0) for p in valid]

    try:
        mzs = [float(x) for x in mzs]
    except Exception:
        mzs = []

    try:
        ints = [float(x) for x in ints]
    except Exception:
        ints = []

    if len(mzs) != len(ints):
        ints = [1.0] * len(mzs)

    return mzs, ints


def prepare_fragments(pmz, mzs, ints, max_fragments):
    pairs = [(float(m), float(i)) for m, i in zip(mzs, ints) if 0 < float(m) < pmz]

    if max_fragments > 0 and len(pairs) > max_fragments:
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:max_fragments]

    pairs = sorted(pairs, key=lambda x: x[0])
    return [m for m, _ in pairs], [i for _, i in pairs]


def load_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        obj = json.loads(text)

        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]

        if isinstance(obj, dict):
            for key in ("data","items","records","spectra","examples"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]

            if "spectrum" in obj or "metadata" in obj:
                return [obj]

    except Exception:
        pass

    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
                if isinstance(x, dict):
                    rows.append(x)
            except Exception:
                pass
    return rows


def parse_limits(text):
    limits = dict(DEFAULT_LIMITS)
    for piece in text.split(","):
        if not piece.strip():
            continue
        e, v = piece.split("=", 1)
        e = e.strip()
        if e not in ATOMIC_MASS:
            raise ValueError(e)
        limits[e] = int(v.strip())
    return limits


# =============================================================================
# Direct rule dynamic import
# =============================================================================

def load_rule_module(rule_file):
    path = Path(rule_file).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("user_formula_rule", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.13 dataclass imports.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    required = [
        "build_candidates",
        "normalize_elements_input",
        "normalize_max_counts_input",
        "normalize_adducts_input",
    ]

    missing = [x for x in required if not hasattr(module, x)]

    if missing:
        raise AttributeError(
            "Rule file missing: " + ", ".join(missing)
        )

    return module


# =============================================================================
# V3 train-derived extension signatures
# =============================================================================

def rare_signature(c):
    return tuple(e for e in FINAL_RARE if c.get(e, 0) > 0)


def build_extension_signatures(all_records, min_train_count, max_signatures):
    counter = Counter()

    for item in all_records:
        if str(get_fold(item)).lower() != "train":
            continue

        f = get_true_formula(item)
        c = parse_formula(f) if f else None

        if not c:
            continue

        if any(e not in FINAL_ELEMENTS for e in c):
            continue

        sig = rare_signature(c)

        if any(e in sig for e in NEW_ELEMENTS):
            counter[sig] += 1

    items = [(sig, n) for sig, n in counter.items() if n >= min_train_count]
    items.sort(key=lambda x: (-x[1], len(x[0]), x[0]))

    if max_signatures > 0:
        items = items[:max_signatures]

    signatures = [sig for sig, _ in items]

    for e in NEW_ELEMENTS:
        if (e,) not in signatures:
            signatures.append((e,))

    seen = set()
    result = []

    for sig in signatures:
        if sig not in seen:
            seen.add(sig)
            result.append(sig)

    return result, counter


# =============================================================================
# Chemistry
# =============================================================================

def calc_dbe(c):
    C = c.get("C",0)
    Si = c.get("Si",0)
    H = c.get("H",0)
    N = c.get("N",0)
    P = c.get("P",0)
    X = sum(c.get(e,0) for e in ("F","Cl","Br","I"))
    return 1 + C + Si - H/2 + (N+P)/2 - X/2


def dbe_valid(d):
    return d >= 0 and abs(d - round(d)) <= 1e-6


def nitrogen_rule_valid(c):
    return nominal_mass(c) % 2 == c.get("N",0) % 2


def ratio_valid(c):
    C = c.get("C",0)
    H = c.get("H",0)

    return (
        C > 0
        and H >= 0
        and 0.20 <= H/C <= 3.20
        and c.get("O",0)/C <= 2.0
        and c.get("N",0)/C <= 1.0
    )


def element_level(c):
    if any(c.get(e,0) > 0 for e in FINAL_RARE):
        return 2
    return 1 if c.get("N",0) > 0 else 0


def legacy_element_penalty(c):
    return (
        0.12*c.get("N",0)
        + 0.28*c.get("P",0)
        + 0.24*c.get("S",0)
        + 0.32*c.get("Cl",0)
        + 0.38*c.get("Br",0)
    )


def final_element_penalty(c):
    return (
        0.10*c.get("N",0)
        + 0.20*c.get("P",0)
        + 0.18*c.get("S",0)
        + 0.08*c.get("F",0)
        + 0.20*c.get("Cl",0)
        + 0.24*c.get("Br",0)
        + 0.16*c.get("I",0)
        + 0.12*c.get("Si",0)
    )


def legacy_iso(c):
    return (
        0.25*c.get("Cl",0)
        + 0.30*c.get("Br",0)
        + 0.08*max(0,c.get("S",0)-1)
    )


def final_iso(c):
    return (
        0.12*c.get("Cl",0)
        + 0.16*c.get("Br",0)
        + 0.04*max(0,c.get("S",0)-1)
        + 0.04*max(0,c.get("Si",0)-1)
    )


def rule_penalty(c, d):
    p = 0.0

    if not dbe_valid(d):
        p += 8

    if not nitrogen_rule_valid(c):
        p += 3

    if not ratio_valid(c):
        p += 2

    if d > 50:
        p += (d - 50) * 0.05

    return p


def carbon_range_old(mass, frags, mode):
    theoretical_max = max(1, int(mass / 12))
    center = max(1, round(mass / 22))
    wf = 0.45 - min(len(frags), 4) * 0.05
    width = max(4, round(center * wf))

    lo = max(1, center - width)
    hi = min(theoretical_max, center + width)

    if frags:
        lo = max(lo, max(1, int(max(frags) / 28)))

    if mode == "wide":
        extra = max(8, int(width * 1.75))
        lo = max(1, lo - extra)
        hi = min(theoretical_max, hi + extra)

    if lo > hi:
        return 1, theoretical_max

    return lo, hi


def carbon_range_v3(mass, frags):
    theoretical_max = max(1, int(mass / 12))
    center = max(1, round(mass / 22))
    wf = 0.45 - min(len(frags), 2) * 0.05
    width = max(4, round(center * wf))
    margin = max(5, round(center * 0.1))

    lo = max(1, center - width - margin)
    hi = min(theoretical_max, center + width + margin)

    if frags:
        lo = max(lo, max(1, int(max(frags) / 28 * 0.6)))

    if lo > hi:
        return 1, theoretical_max

    return lo, hi


# =============================================================================
# Enumerators
# =============================================================================

def enumerate_chno(target, ppm, limits, cr):
    tol = target * ppm / 1e6
    loC, hiC = cr
    out = []

    max_o = min(limits["O"], int(target / ATOMIC_MASS["O"]) + 1)

    for O in range(max_o + 1):
        mass_o = O * ATOMIC_MASS["O"]

        max_n = min(
            limits["N"],
            int(max(0.0, target - mass_o) / ATOMIC_MASS["N"]) + 1,
        )

        for N in range(max_n + 1):
            mass_on = mass_o + N * ATOMIC_MASS["N"]

            maxC = min(
                hiC,
                int((target - mass_on + tol) / 12),
            )

            for C in range(loC, maxC + 1):
                rem = target - (mass_on + C * 12)

                if rem < -tol:
                    break

                H = round(rem / ATOMIC_MASS["H"])

                if H < 0 or H > limits["H"]:
                    continue

                c = {"C": C, "H": H}

                if O:
                    c["O"] = O

                if N:
                    c["N"] = N

                mass = exact_mass(c)

                if abs(mass - target) > tol:
                    continue

                d = calc_dbe(c)

                if dbe_valid(d):
                    out.append((c, mass, d))

    return out


def enumerate_legacy_8(target, ppm, limits, cr):
    tol = target * ppm / 1e6
    variables = ["Br","Cl","S","P","O","N"]
    loC, hiC = cr
    out = []

    def rec(i, counts, mass_so_far):
        if target - mass_so_far < loC * 12 - tol:
            return

        if i == len(variables):
            maxC = min(
                hiC,
                int((target - mass_so_far + tol) / 12),
            )

            for C in range(loC, maxC + 1):
                rem = target - (mass_so_far + C * 12)

                if rem < -tol:
                    break

                H = round(rem / ATOMIC_MASS["H"])

                if H < 0 or H > limits["H"]:
                    continue

                c = dict(counts)
                c["C"] = C
                c["H"] = H

                mass = exact_mass(c)

                if abs(mass - target) > tol:
                    continue

                out.append((c, mass, calc_dbe(c)))

            return

        e = variables[i]
        em = ATOMIC_MASS[e]

        mx = min(
            limits[e],
            int(max(0.0, target - mass_so_far - loC*12 + tol) / em),
        )

        for n in range(mx + 1):
            if n:
                counts[e] = n
            else:
                counts.pop(e, None)

            rec(i + 1, counts, mass_so_far + n * em)

        counts.pop(e, None)

    rec(0, {}, 0.0)
    return out


def enumerate_extension(target, ppm, signature, limits, cr):
    tol = target * ppm / 1e6
    loC, hiC = cr

    variables = list(signature) + ["O","N"]

    minimum = {
        e: (1 if e in signature else 0)
        for e in variables
    }

    maximum = {
        e: min(limits[e], int(target / ATOMIC_MASS[e]) + 1)
        for e in variables
    }

    suffix = [0.0] * (len(variables) + 1)

    for i in range(len(variables)-1, -1, -1):
        e = variables[i]
        suffix[i] = suffix[i+1] + minimum[e] * ATOMIC_MASS[e]

    out = []

    def rec(i, counts, mass_so_far):
        if mass_so_far + suffix[i] + loC*12 > target + tol:
            return

        if i == len(variables):
            maxC = min(
                hiC,
                int((target - mass_so_far + tol) / 12),
            )

            for C in range(loC, maxC + 1):
                rem = target - (mass_so_far + C*12)

                if rem < -tol:
                    break

                H = round(rem / ATOMIC_MASS["H"])

                if H < 0 or H > limits["H"]:
                    continue

                c = dict(counts)
                c["C"] = C
                c["H"] = H

                mass = exact_mass(c)

                if abs(mass - target) > tol:
                    continue

                d = calc_dbe(c)

                if dbe_valid(d):
                    out.append((c, mass, d))

            return

        e = variables[i]
        em = ATOMIC_MASS[e]

        future_min = suffix[i+1] + loC*12

        max_by_mass = int(
            max(0.0, target - mass_so_far - future_min + tol) / em
        )

        lo = minimum[e]
        hi = min(maximum[e], max_by_mass)

        for n in range(lo, hi + 1):
            if n:
                counts[e] = n
            else:
                counts.pop(e, None)

            rec(i + 1, counts, mass_so_far + n*em)

        counts.pop(e, None)

    rec(0, {}, 0.0)
    return out


# =============================================================================
# Candidate ranking
# =============================================================================

def make_candidate(c, mass, d, pmz, adduct, final):
    calc_mz = mass + adduct.mass_shift
    ppm = (calc_mz - pmz) / pmz * 1e6

    nr = nitrogen_rule_valid(c)
    rv = ratio_valid(c)

    return Candidate(
        formula=format_formula(c),
        adduct=adduct.name,
        ppm_error=ppm,
        dbe=d,
        counts=c,
        element_level=element_level(c),
        rule_valid=dbe_valid(d) and nr and rv,
        nitrogen_rule_valid=nr,
        ratio_valid=rv,
        isotope_prior_penalty=final_iso(c) if final else legacy_iso(c),
        adduct_prior_penalty=adduct.prior_penalty,
    )


def original_key(c, polarity, final):
    pri = {a.name: a.priority for a in ADDUCTS[polarity]}
    elem_pen = final_element_penalty(c.counts) if final else legacy_element_penalty(c.counts)

    return (
        c.element_level,
        pri.get(c.adduct, 99),
        abs(c.ppm_error) + elem_pen,
        abs(c.ppm_error),
    )


def rule_key(c):
    return (
        rule_penalty(c.counts, c.dbe),
        abs(c.ppm_error),
        c.adduct_prior_penalty,
    )


def prior_key(c, final):
    elem_pen = final_element_penalty(c.counts) if final else legacy_element_penalty(c.counts)

    return (
        c.isotope_prior_penalty + elem_pen,
        abs(c.ppm_error),
        c.adduct_prior_penalty,
    )


def integrated_score(c, source_count, final):
    elem_pen = final_element_penalty(c.counts) if final else legacy_element_penalty(c.counts)

    return (
        0.80*abs(c.ppm_error)
        + 1.20*rule_penalty(c.counts, c.dbe)
        + 1.00*c.adduct_prior_penalty
        + 0.70*c.isotope_prior_penalty
        + 0.30*elem_pen
        - 0.80*source_count
    )


def unique_best(candidates, key):
    forms = []
    best = {}
    seen = set()

    for c in sorted(candidates, key=key):
        if c.formula in seen:
            continue

        seen.add(c.formula)
        forms.append(c.formula)
        best[c.formula] = c

    return forms, best


def union_ranked(branches, source_top, final):
    by_formula = defaultdict(list)
    sources = defaultdict(set)

    for source, (forms, best) in branches.items():
        selected = forms if source_top <= 0 else forms[:source_top]

        for f in selected:
            if f not in best:
                continue

            by_formula[f].append(best[f])
            sources[f].add(source)

    scored = []

    for f, candidates in by_formula.items():
        source_count = len(sources[f])

        score = min(
            integrated_score(c, source_count, final)
            for c in candidates
        )

        scored.append((score, f))

    scored.sort(key=lambda x: (x[0], x[1]))

    return [f for _, f in scored]


# =============================================================================
# OLD / V3 / DIRECT
# =============================================================================

def build_old(pmz, polarity, frags, limits, args):
    original = []
    expanded = []

    for adduct in ADDUCTS[polarity]:
        neutral = pmz - adduct.mass_shift

        if neutral <= 0:
            continue

        cr_orig = carbon_range_old(neutral, frags, "original")

        for c, mass, d in enumerate_chno(
            neutral,
            args.old_ppm,
            limits,
            cr_orig,
        ):
            original.append(
                make_candidate(c, mass, d, pmz, adduct, False)
            )

        cr_wide = carbon_range_old(neutral, frags, "wide")

        for c, mass, d in enumerate_legacy_8(
            neutral,
            args.old_ppm,
            limits,
            cr_wide,
        ):
            expanded.append(
                make_candidate(c, mass, d, pmz, adduct, False)
            )

    strict = [c for c in expanded if c.rule_valid]

    of, ob = unique_best(
        original,
        lambda c: original_key(c, polarity, False),
    )

    rf, rb = unique_best(
        strict,
        rule_key,
    )

    pf, pb = unique_best(
        strict,
        lambda c: prior_key(c, False),
    )

    return union_ranked(
        {
            "original": (of, ob),
            "rule": (rf, rb),
            "prior": (pf, pb),
        },
        args.old_source_top,
        False,
    )


def build_v3(pmz, polarity, frags, limits, args, signatures):
    original = []
    expanded = []

    for adduct in ADDUCTS[polarity]:
        neutral = pmz - adduct.mass_shift

        if neutral <= 0:
            continue

        cr = carbon_range_v3(neutral, frags)

        for c, mass, d in enumerate_chno(
            neutral,
            args.v3_ppm,
            limits,
            cr,
        ):
            original.append(
                make_candidate(c, mass, d, pmz, adduct, True)
            )

        for c, mass, d in enumerate_legacy_8(
            neutral,
            args.v3_ppm,
            limits,
            cr,
        ):
            expanded.append(
                make_candidate(c, mass, d, pmz, adduct, True)
            )

        for sig in signatures:
            for c, mass, d in enumerate_extension(
                neutral,
                args.v3_ppm,
                sig,
                limits,
                cr,
            ):
                expanded.append(
                    make_candidate(c, mass, d, pmz, adduct, True)
                )

    strict = [c for c in expanded if c.rule_valid]

    of, ob = unique_best(
        original,
        lambda c: original_key(c, polarity, True),
    )

    rf, rb = unique_best(
        strict,
        rule_key,
    )

    pf, pb = unique_best(
        strict,
        lambda c: prior_key(c, True),
    )

    return union_ranked(
        {
            "original": (of, ob),
            "rule": (rf, rb),
            "prior": (pf, pb),
        },
        args.v3_source_top,
        True,
    )


def build_direct(rule, pmz, polarity, mzs, ints, args):
    direct_frags, _ = prepare_fragments(
        pmz,
        mzs,
        ints,
        args.direct_max_fragments,
    )

    elements = rule.normalize_elements_input("C,H,N,O")
    limits = rule.normalize_max_counts_input("")
    adducts = rule.normalize_adducts_input(polarity, [])

    candidates = rule.build_candidates(
        precursor_mz=pmz,
        polarity=polarity,
        adducts=adducts,
        fragments=direct_frags,
        ppm=args.direct_ppm,
        allowed_elements=elements,
        raw_limits=limits,
        use_dbe_filter=True,
    )

    out = []
    seen = set()

    for c in candidates:
        key = formula_key(c.counts)

        if key is None or key in seen:
            continue

        seen.add(key)
        out.append(format_formula(c.counts))

    return out


# =============================================================================
# Final union
# =============================================================================

def ordered_union(*lists):
    out = []
    seen = set()

    for values in lists:
        for f in values:
            key = formula_key(f)

            if key is None or key in seen:
                continue

            seen.add(key)
            out.append(f)

    return out


def contains_true(formulas, true_formula):
    true_key = formula_key(true_formula)
    return int(
        true_key in {formula_key(f) for f in formulas}
    )


# =============================================================================
# Worker
# =============================================================================

class SampleTimeout(Exception):
    pass


def alarm_handler(signum, frame):
    raise SampleTimeout()


RULE = None
ARGS = None
LIMITS = None
SIGNATURES = None


def init_worker(rule_file, args_dict, limits, signatures):
    global RULE, ARGS, LIMITS, SIGNATURES

    RULE = load_rule_module(rule_file)
    ARGS = argparse.Namespace(**args_dict)
    LIMITS = limits
    SIGNATURES = list(signatures)


def zero_result(idx, item, reason):
    true_formula = get_true_formula(item)
    true_counts = parse_formula(true_formula) if true_formula else {}

    return {
        "idx": idx,
        "identifier": get_identifier(item, idx),
        "true_formula": true_formula,
        "has_F": int(bool(true_counts) and true_counts.get("F",0) > 0),
        "has_I": int(bool(true_counts) and true_counts.get("I",0) > 0),
        "has_Si": int(bool(true_counts) and true_counts.get("Si",0) > 0),
        "has_new_element": int(
            bool(true_counts)
            and any(true_counts.get(e,0) > 0 for e in NEW_ELEMENTS)
        ),
        "old_hit": 0,
        "v3_hit": 0,
        "direct_hit": 0,
        "old_v3_hit": 0,
        "final_hit": 0,
        "old_size": 0,
        "v3_size": 0,
        "direct_size": 0,
        "old_v3_size": 0,
        "final_size": 0,
        "old_short": [],
        "v3_short": [],
        "direct_short": [],
        "old_v3_union": [],
        "final_union": [],
        "status": reason,
    }


def worker(payload):
    idx, item = payload

    if ARGS.sample_timeout > 0:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(ARGS.sample_timeout)

    try:
        true_formula = get_true_formula(item)
        pmz = get_precursor_mz(item)
        polarity = get_polarity(item)

        if not true_formula or pmz is None or not polarity:
            return idx, zero_result(idx, item, "skip"), "skip"

        mzs, ints = get_fragments_and_intensities(item)

        heuristic_frags, _ = prepare_fragments(
            pmz,
            mzs,
            ints,
            ARGS.heuristic_max_fragments,
        )

        old_full = build_old(
            pmz,
            polarity,
            heuristic_frags,
            LIMITS,
            ARGS,
        )

        v3_full = build_v3(
            pmz,
            polarity,
            heuristic_frags,
            LIMITS,
            ARGS,
            SIGNATURES,
        )

        direct_full = build_direct(
            RULE,
            pmz,
            polarity,
            mzs,
            ints,
            ARGS,
        )

        old_short = old_full[:ARGS.old_short]
        v3_short = v3_full[:ARGS.v3_short]
        direct_short = direct_full[:ARGS.direct_short]

        old_v3_union = ordered_union(
            v3_short,
            old_short,
        )

        final_union = ordered_union(
            v3_short,
            old_short,
            direct_short,
        )

        true_counts = parse_formula(true_formula) or {}

        result = {
            "idx": idx,
            "identifier": get_identifier(item, idx),
            "true_formula": true_formula,

            "has_F": int(true_counts.get("F",0) > 0),
            "has_I": int(true_counts.get("I",0) > 0),
            "has_Si": int(true_counts.get("Si",0) > 0),
            "has_new_element": int(
                any(true_counts.get(e,0) > 0 for e in NEW_ELEMENTS)
            ),

            "old_hit": contains_true(old_short, true_formula),
            "v3_hit": contains_true(v3_short, true_formula),
            "direct_hit": contains_true(direct_short, true_formula),
            "old_v3_hit": contains_true(old_v3_union, true_formula),
            "final_hit": contains_true(final_union, true_formula),

            "old_size": len(old_short),
            "v3_size": len(v3_short),
            "direct_size": len(direct_short),
            "old_v3_size": len(old_v3_union),
            "final_size": len(final_union),

            "old_short": old_short,
            "v3_short": v3_short,
            "direct_short": direct_short,
            "old_v3_union": old_v3_union,
            "final_union": final_union,

            "status": "ok",
        }

        return idx, result, "ok"

    except SampleTimeout:
        return idx, zero_result(idx, item, "timeout"), "timeout"

    except Exception as e:
        row = zero_result(idx, item, f"error:{repr(e)}")
        return idx, row, "error"

    finally:
        if ARGS.sample_timeout > 0:
            signal.alarm(0)


# =============================================================================
# Aggregation
# =============================================================================

def percentile(values, q):
    if not values:
        return math.nan

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    pos = (len(values)-1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return float(values[lo])

    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def aggregate(rows, subset=None):
    selected = rows

    if subset is not None:
        selected = [r for r in rows if r.get(subset)]

    n = len(selected)

    if n == 0:
        return {
            "Subset": subset or "ALL",
            "N": 0,
        }

    def rate(field):
        return sum(r[field] for r in selected) / n

    final_sizes = [r["final_size"] for r in selected]

    return {
        "Subset": subset or "ALL",
        "N": n,

        "OLD_Top10_Coverage": rate("old_hit"),
        "V3_Top10_Coverage": rate("v3_hit"),
        "DIRECT_Top5_Coverage": rate("direct_hit"),
        "OLD_V3_Union_Coverage": rate("old_v3_hit"),
        "FINAL_Union_Coverage": rate("final_hit"),

        "OLD_Top10_Hits": sum(r["old_hit"] for r in selected),
        "V3_Top10_Hits": sum(r["v3_hit"] for r in selected),
        "DIRECT_Top5_Hits": sum(r["direct_hit"] for r in selected),
        "OLD_V3_Union_Hits": sum(r["old_v3_hit"] for r in selected),
        "FINAL_Union_Hits": sum(r["final_hit"] for r in selected),

        "Avg_Final_Candidates": sum(final_sizes) / n,
        "Median_Final_Candidates": statistics.median(final_sizes),
        "P90_Final_Candidates": percentile(final_sizes, 0.90),
        "P95_Final_Candidates": percentile(final_sizes, 0.95),
        "Max_Final_Candidates": max(final_sizes),
    }


def marginal_counts(rows):
    add_old = sum(
        (not r["v3_hit"]) and r["old_v3_hit"]
        for r in rows
    )

    add_direct = sum(
        (not r["old_v3_hit"]) and r["final_hit"]
        for r in rows
    )

    direct_vs_v3 = sum(
        (not r["v3_hit"]) and r["direct_hit"]
        for r in rows
    )

    old_vs_v3 = sum(
        (not r["v3_hit"]) and r["old_hit"]
        for r in rows
    )

    final_misses = sum(
        not r["final_hit"]
        for r in rows
    )

    return {
        "OLD_rescues_beyond_V3": add_old,
        "DIRECT_rescues_beyond_OLD_V3": add_direct,
        "DIRECT_rescues_beyond_V3": direct_vs_v3,
        "OLD_rescues_beyond_V3_individual": old_vs_v3,
        "FINAL_remaining_misses": final_misses,
    }


def save_summary_csv(summary_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    seen = set()

    for row in summary_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--rule-file", required=True)

    parser.add_argument("--fold", default="test")
    parser.add_argument("--max-samples", type=int, default=2000)

    parser.add_argument("--old-ppm", type=float, default=5.0)
    parser.add_argument("--v3-ppm", type=float, default=10.0)
    parser.add_argument("--direct-ppm", type=float, default=10.0)

    parser.add_argument("--old-source-top", type=int, default=200)
    parser.add_argument("--v3-source-top", type=int, default=200)

    parser.add_argument("--old-short", type=int, default=10)
    parser.add_argument("--v3-short", type=int, default=10)
    parser.add_argument("--direct-short", type=int, default=5)

    parser.add_argument(
        "--max-counts",
        default="C=80,H=160,N=20,O=30,P=6,S=8,F=30,Cl=6,Br=4,I=3,Si=4",
    )

    parser.add_argument("--extension-min-train-count", type=int, default=1)
    parser.add_argument("--max-extension-signatures", type=int, default=32)

    parser.add_argument("--heuristic-max-fragments", type=int, default=50)
    parser.add_argument(
        "--direct-max-fragments",
        type=int,
        default=0,
        help="0 = all MS2 peaks for the direct rule.",
    )

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-timeout", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=20)

    parser.add_argument(
        "--output-csv",
        default="./outputs/final_candidate_union_test2000_summary.csv",
    )

    parser.add_argument(
        "--checkpoint-jsonl",
        default="./outputs/final_candidate_union_test2000_details.jsonl",
    )

    parser.add_argument("--print-signatures", action="store_true")

    args = parser.parse_args()

    # Validate direct rule import.
    _ = load_rule_module(args.rule_file)

    all_records = load_records(args.input)

    signatures, sig_counts = build_extension_signatures(
        all_records,
        args.extension_min_train_count,
        args.max_extension_signatures,
    )

    records = all_records

    if args.fold:
        records = [
            r for r in records
            if str(get_fold(r)).lower() == args.fold.lower()
        ]

    if args.max_samples > 0:
        records = records[:args.max_samples]

    limits = parse_limits(args.max_counts)

    print("="*88)
    print("FINAL PRE-LLM CANDIDATE UNION")
    print("="*88)
    print(f"Records                  : {len(records)}")
    print(f"Fold                     : {args.fold}")
    print()
    print(f"V3 main branch           : Top-{args.v3_short}, 10 ppm, 11 elements")
    print(f"OLD complement branch    : Top-{args.old_short}, 5 ppm, 8 elements")
    print(f"DIRECT fragment branch   : Top-{args.direct_short}, 10 ppm, CHNO")
    print(f"Maximum final candidates : {args.v3_short + args.old_short + args.direct_short}")
    print()
    print(f"Extension signatures     : {len(signatures)}")
    print(f"Workers                  : {args.num_workers}")
    print(f"Sample timeout           : {args.sample_timeout}s")
    print("LLM                      : OFF")
    print("SIRIUS rescue            : OFF")
    print("Exact annotated adduct   : NOT USED")
    print("="*88, flush=True)

    if args.print_signatures:
        print("\nF/I/Si extension signatures learned from TRAIN:")
        for i, sig in enumerate(signatures, 1):
            print(
                f"{i:02d}. {sig} "
                f"train_count={sig_counts.get(sig,0)}"
            )

    rows = []
    timeout_count = 0
    error_count = 0
    skip_count = 0

    checkpoint_path = Path(args.checkpoint_jsonl)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = list(enumerate(records, 1))

    with checkpoint_path.open("w", encoding="utf-8") as ckpt:
        with mp.get_context("fork").Pool(
            processes=min(args.num_workers, max(1, len(tasks))),
            initializer=init_worker,
            initargs=(
                args.rule_file,
                vars(args).copy(),
                limits,
                signatures,
            ),
        ) as pool:

            for done, (idx, row, status) in enumerate(
                pool.imap_unordered(worker, tasks, chunksize=1),
                1,
            ):
                rows.append(row)

                ckpt.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                ckpt.flush()

                if status == "timeout":
                    timeout_count += 1
                    print(
                        f"[WARN] timeout record {idx}; counted as miss",
                        file=sys.stderr,
                        flush=True,
                    )

                elif status == "error":
                    error_count += 1
                    print(
                        f"[WARN] error record {idx}; counted as miss",
                        file=sys.stderr,
                        flush=True,
                    )

                elif status == "skip":
                    skip_count += 1

                if args.progress_every > 0 and done % args.progress_every == 0:
                    print(
                        f"Processed {done}/{len(tasks)} | "
                        f"timeout={timeout_count} error={error_count} skip={skip_count}",
                        file=sys.stderr,
                        flush=True,
                    )

    print("\n" + "="*88)
    print("RESULT")
    print("="*88)
    print(f"N         : {len(rows)}")
    print(f"Timeouts  : {timeout_count}")
    print(f"Errors    : {error_count}")
    print(f"Skipped   : {skip_count}")

    summary = [
        aggregate(rows),
        aggregate(rows, "has_new_element"),
        aggregate(rows, "has_F"),
        aggregate(rows, "has_I"),
        aggregate(rows, "has_Si"),
    ]

    cols = [
        "Subset","N",
        "OLD_Top10_Coverage",
        "V3_Top10_Coverage",
        "DIRECT_Top5_Coverage",
        "OLD_V3_Union_Coverage",
        "FINAL_Union_Coverage",
        "Avg_Final_Candidates",
        "Median_Final_Candidates",
        "P90_Final_Candidates",
        "P95_Final_Candidates",
        "Max_Final_Candidates",
    ]

    print("\n|" + "|".join(cols) + "|")
    print("|" + "|".join(["---"]*len(cols)) + "|")

    for row in summary:
        vals = []

        for col in cols:
            v = row.get(col, "")

            if isinstance(v, float):
                if "Coverage" in col:
                    vals.append(f"{v*100:.2f}%")
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))

        print("|" + "|".join(vals) + "|")

    marginal = marginal_counts(rows)

    print("\nMARGINAL CONTRIBUTIONS")
    print("-"*88)
    print(
        f"OLD rescues beyond V3               : "
        f"{marginal['OLD_rescues_beyond_V3']}"
    )
    print(
        f"DIRECT rescues beyond V3            : "
        f"{marginal['DIRECT_rescues_beyond_V3']}"
    )
    print(
        f"DIRECT rescues beyond OLD+V3        : "
        f"{marginal['DIRECT_rescues_beyond_OLD_V3']}"
    )
    print(
        f"FINAL remaining candidate misses    : "
        f"{marginal['FINAL_remaining_misses']}"
    )

    save_summary_csv(
        summary,
        args.output_csv,
    )

    print(f"\nSaved summary : {Path(args.output_csv).resolve()}")
    print(f"Saved details : {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
