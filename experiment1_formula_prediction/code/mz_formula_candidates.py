#!/usr/bin/env python
"""Enumerate molecular-formula candidates from precursor m/z.

Edit the values in ``RUN_CONFIG`` below, then run:

    python mz_formula_candidates.py

This script is designed for LC-MS/MS annotation when you know:
1. precursor m/z
2. polarity (positive / negative)
3. optional MS2 fragment m/z values

The exact precursor adduct can be unknown. In this case the script tries a
small set of common adducts for the selected polarity and enumerates candidate
neutral molecular formulas for each adduct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


ATOMIC_MASS: Dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "F":18.998403163,
    "I":126.904473,
}

PROTON_MASS = 1.007276466812

DEFAULT_LIMITS: Dict[str, int] = {
    "C": 80,
    "H": 160,
    "N": 20,
    "O": 30,
    "P": 6,
    "S": 8,
    "Cl": 6,
    "Br": 4,
    "F": 30,
    "I":3,
}

ELEMENT_PRINT_ORDER = ("C", "H", "N", "O", "P", "S", "Cl", "Br","F","I")
HETERO_SEARCH_ORDER = ("Br", "Cl", "S", "P", "O", "N","F","I")

COMMON_NEUTRAL_LOSSES: Dict[str, float] = {
    "H2O": 18.010565,
    "NH3": 17.026549,
    "CO": 27.994915,
    "CO2": 43.989830,
    "CH2O": 30.010565,
    "CH3OH": 32.026215,
    "HCOOH": 46.005479,
    "CH3COOH": 60.021129,
    "C6H10O5": 162.052824,
    "C6H12O6": 180.063388,
    "SO3": 79.956815,
    "H3PO4": 97.976896,
}

RUN_CONFIG: Dict[str, object] = {
    
    "precursor_m/z":323.1051,
    "polarity": "positive",
    "fragments":  [65.038597, 79.021141, 80.049393, 92.049423, 92.054985, 92.063148, 93.05732, 108.044205, 110.060066, 119.060394, 131.060364, 135.055313, 149.051224, 158.071411, 161.051071, 174.066284, 177.04599, 188.061996, 189.04628, 203.048996, 216.056778, 234.067139, 250.077591, 267.104858, 268.087891, 277.088745, 278.072784, 295.098999, 295.17807, 323.105072, 323.19519, 324.108154],
    "ppm": 10.0,
    "elements": ["C", "H", "N", "O"],
    "max_counts": {},
    "adducts": [],
    "top": 200,
    "use_dbe_filter": True,
}


@dataclass(frozen=True)
class Adduct:
    name: str
    mass_shift: float
    priority: int


@dataclass
class Candidate:
    adduct: str
    formula: str
    neutral_mass: float
    calculated_mz: float
    ppm_error: float
    dbe: float
    counts: Dict[str, int]
    loss_matches: List[str]
    explained_fragments: int
    ranking_score: float
    element_level: int  # 0=CHO,1=CHON,2=OTHER


ADDUCTS: Dict[str, List[Adduct]] = {
    "positive": [
        Adduct("[M+H]+", PROTON_MASS, 1),
        Adduct("[M+Na]+", 22.989218, 2),
        Adduct("[M+NH4]+", 18.033823, 3),
        Adduct("[M+K]+", 38.963158, 4),
    ],
    "negative": [
        Adduct("[M-H]-", -PROTON_MASS, 1),
        Adduct("[M+Cl]-", 34.969402, 2),
        Adduct("[M+FA-H]-", 44.998201, 3),
        Adduct("[M+Ac-H]-", 59.013851, 4),
    ],
}


def normalize_polarity(value: str) -> str:
    value = value.lower().strip()
    if value in {"positive", "pos", "+"}:
        return "positive"
    if value in {"negative", "neg", "-"}:
        return "negative"
    raise ValueError(f"Unsupported polarity: {value}")


def parse_float_csv(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_elements(text: str) -> List[str]:
    elements = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in ATOMIC_MASS:
            raise ValueError(f"Unsupported element: {item}")
        elements.append(item)
    if "C" not in elements or "H" not in elements:
        raise ValueError("Elements must include at least C and H.")
    seen = set()
    ordered = []
    for element in ELEMENT_PRINT_ORDER:
        if element in elements and element not in seen:
            ordered.append(element)
            seen.add(element)
    return ordered


def parse_max_counts(text: str) -> Dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    if not text.strip():
        return limits
    for piece in text.split(","):
        item = piece.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid max-count item: {item}")
        element, raw_value = item.split("=", 1)
        element = element.strip()
        if element not in ATOMIC_MASS:
            raise ValueError(f"Unsupported element in max-counts: {element}")
        limits[element] = int(raw_value.strip())
    return limits


def normalize_elements_input(value: object) -> List[str]:
    if isinstance(value, str):
        return parse_elements(value)
    if isinstance(value, (list, tuple)):
        return parse_elements(",".join(str(item) for item in value))
    raise ValueError("elements must be a comma-separated string or a list of element symbols.")


def normalize_max_counts_input(value: object) -> Dict[str, int]:
    if value is None:
        return dict(DEFAULT_LIMITS)
    if isinstance(value, str):
        return parse_max_counts(value)
    if isinstance(value, dict):
        limits = dict(DEFAULT_LIMITS)
        for element, count in value.items():
            if element not in ATOMIC_MASS:
                raise ValueError(f"Unsupported element in max_counts: {element}")
            limits[element] = int(count)
        return limits
    raise ValueError("max_counts must be a string like 'C=60,N=4' or a dict.")


def normalize_adducts_input(polarity: str, value: object) -> List[Adduct]:
    if value is None:
        return build_adduct_list(polarity, "")
    if isinstance(value, str):
        return build_adduct_list(polarity, value)
    if isinstance(value, (list, tuple)):
        return build_adduct_list(polarity, ",".join(str(item) for item in value))
    raise ValueError("adducts must be a comma-separated string or a list of adduct names.")


def build_adduct_list(polarity: str, requested: str) -> List[Adduct]:
    adducts = list(ADDUCTS[polarity])
    if not requested.strip():
        return adducts
    requested_names = {item.strip() for item in requested.split(",") if item.strip()}
    filtered = [adduct for adduct in adducts if adduct.name in requested_names]
    if not filtered:
        allowed = ", ".join(adduct.name for adduct in adducts)
        raise ValueError(f"No requested adduct matched. Allowed for {polarity}: {allowed}")
    return filtered


def format_formula(counts: Dict[str, int]) -> str:
    parts: List[str] = []
    for element in ELEMENT_PRINT_ORDER:
        count = counts.get(element, 0)
        if count <= 0:
            continue
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def exact_mass(counts: Dict[str, int]) -> float:
    return sum(ATOMIC_MASS[element] * count for element, count in counts.items())


def calc_dbe(counts: Dict[str, int]) -> float:
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    p = counts.get("P", 0)
    halogens = counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0) + counts.get("I", 0)
    return 1.0 + c - h / 2.0 + (n + p) / 2.0 - halogens / 2.0


def is_plausible_formula(counts: Dict[str, int], use_dbe_filter: bool) -> Tuple[bool, float]:
    if counts.get("C", 0) < 1 or counts.get("H", 0) < 0:
        return False, 0.0
    dbe = calc_dbe(counts)
    if not use_dbe_filter:
        return True, dbe
    if dbe < 0:
        return False, dbe
    if abs(dbe - round(dbe)) > 1e-6:
        return False, dbe
    return True, dbe


def mass_tolerance_da(mass: float, ppm: float) -> float:
    return mass * ppm / 1_000_000.0


def explain_neutral_losses(precursor_mz: float, fragments: Sequence[float], ppm: float) -> List[str]:
    notes: List[str] = []
    for fragment in fragments:
        if fragment >= precursor_mz:
            continue
        loss = precursor_mz - fragment
        tol = mass_tolerance_da(loss, ppm)
        matches = []
        for name, ref_mass in COMMON_NEUTRAL_LOSSES.items():
            if abs(loss - ref_mass) <= tol:
                matches.append(name)
        if matches:
            joined = "/".join(matches)
            notes.append(f"{fragment:.4f} (loss {loss:.4f}: {joined})")
    return notes


def effective_limits(target_mass: float, allowed_elements: Sequence[str], raw_limits: Dict[str, int]) -> Dict[str, int]:
    limits = dict(raw_limits)
    for element in allowed_elements:
        natural_max = int(target_mass / ATOMIC_MASS[element]) + 1
        limits[element] = min(limits.get(element, natural_max), natural_max)
    return limits


def estimate_carbon_range(target_mass: float, fragments: Sequence[float]) -> Tuple[int, int]:
    theoretical_max = max(1, int(target_mass / ATOMIC_MASS["C"]))
    center = max(1, round(target_mass / 22.0))
    fragment_count = len(fragments)
    width_factor = 0.45 - min(fragment_count, 2) * 0.05
    width = max(4, round(center * width_factor))

    c_min = max(1, center - width - max(5, round(center * 0.1)))
    c_max = min(theoretical_max, center + width + max(5, round(center * 0.1)))

    if fragments:
        largest_fragment = max(fragments)
        fragment_floor = max(1, int(largest_fragment / 28.0 * 0.6))
        c_min = max(c_min, fragment_floor)

    if c_min > c_max:
        c_min = max(1, min(c_min, theoretical_max))
        c_max = max(c_min, theoretical_max)

    return c_min, c_max


def is_subformula(child: Dict[str, int], parent: Dict[str, int]) -> bool:
    return all(child.get(element, 0) <= parent.get(element, 0) for element in ELEMENT_PRINT_ORDER)


def enumerate_formulas(
    target_mass: float,
    ppm: float,
    allowed_elements: Sequence[str],
    limits: Dict[str, int],
    use_dbe_filter: bool,
    carbon_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[Dict[str, int], float, float]]:
    tol = mass_tolerance_da(target_mass, ppm)
    allowed_hetero = [element for element in HETERO_SEARCH_ORDER if element in allowed_elements]
    results: List[Tuple[Dict[str, int], float, float]] = []
    max_h = limits.get("H", DEFAULT_LIMITS["H"])
    max_c = limits.get("C", DEFAULT_LIMITS["C"])
    min_c = 1
    if carbon_range is not None:
        min_c = max(1, carbon_range[0])
        max_c = min(max_c, carbon_range[1])

    def backtrack(index: int, counts: Dict[str, int], mass_so_far: float) -> None:
        remaining_after_current = target_mass - mass_so_far
        if remaining_after_current < ATOMIC_MASS["C"] - tol:
            return

        if index == len(allowed_hetero):
            max_c_here = min(max_c, int((target_mass - mass_so_far + tol) / ATOMIC_MASS["C"]))
            if max_c_here < min_c:
                return
            for c_count in range(min_c, max_c_here + 1):
                c_mass = mass_so_far + c_count * ATOMIC_MASS["C"]
                remaining = target_mass - c_mass
                if remaining < -tol:
                    break
                h_count = round(remaining / ATOMIC_MASS["H"])
                if h_count < 0 or h_count > max_h:
                    continue
                final_counts = dict(counts)
                final_counts["C"] = c_count
                final_counts["H"] = h_count
                calc_mass = exact_mass(final_counts)
                if abs(calc_mass - target_mass) > tol:
                    continue
                ok, dbe = is_plausible_formula(final_counts, use_dbe_filter)
                if not ok:
                    continue
                results.append((final_counts, calc_mass, dbe))
            return

        element = allowed_hetero[index]
        element_mass = ATOMIC_MASS[element]
        max_count = min(limits[element], int((target_mass - mass_so_far + tol) / element_mass))

        for count in range(max_count + 1):
            if count:
                counts[element] = count
            elif element in counts:
                counts.pop(element)
            new_mass = mass_so_far + count * element_mass
            backtrack(index + 1, counts, new_mass)

        counts.pop(element, None)

    backtrack(0, {}, 0.0)
    return results


def fragment_mass_targets(fragment_mz: float, polarity: str) -> List[float]:
    if polarity == "positive":
        candidates = [fragment_mz - PROTON_MASS, fragment_mz]
    else:
        candidates = [fragment_mz + PROTON_MASS, fragment_mz]
    return [mass for mass in candidates if mass > 0]


def build_fragment_formula_library(
    fragments: Sequence[float],
    polarity: str,
    ppm: float,
    allowed_elements: Sequence[str],
    raw_limits: Dict[str, int],
) -> Dict[float, List[Dict[str, int]]]:
    library: Dict[float, List[Dict[str, int]]] = {}
    for fragment in fragments:
        formulas_by_text: Dict[str, Dict[str, int]] = {}
        for target_mass in fragment_mass_targets(fragment, polarity):
            limits = effective_limits(target_mass, allowed_elements, raw_limits)
            matches = enumerate_formulas(
                target_mass=target_mass,
                ppm=ppm,
                allowed_elements=allowed_elements,
                limits=limits,
                use_dbe_filter=True,
                carbon_range=estimate_carbon_range(target_mass, []),
            )
            for counts, _, _ in matches:
                formulas_by_text[format_formula(counts)] = counts
        library[fragment] = list(formulas_by_text.values())
    return library


def count_explained_fragments(
    parent_counts: Dict[str, int],
    fragment_library: Dict[float, List[Dict[str, int]]],
) -> int:
    explained = 0
    for options in fragment_library.values():
        if any(is_subformula(option, parent_counts) for option in options):
            explained += 1
    return explained


def hetero_penalty(counts: Dict[str, int]) -> float:
    return (
        0.6 * counts.get("N", 0)
        + 0.9 * counts.get("P", 0)
        + 0.8 * counts.get("S", 0)
        + 0.8 * (counts.get("Cl", 0) + counts.get("Br", 0))
    )


# ====================== 核心：元素类型判断 ======================
def get_element_level(counts: Dict[str, int]) -> int:
    has_hetero = (
        counts.get("S", 0) > 0
        or counts.get("P", 0) > 0
        or counts.get("Cl", 0) > 0
        or counts.get("Br", 0) > 0
    )
    if has_hetero:
        return 2
    if counts.get("N", 0) > 0:
        return 1
    return 0


def build_candidates(
    precursor_mz: float,
    polarity: str,
    adducts: Sequence[Adduct],
    fragments: Sequence[float],
    ppm: float,
    allowed_elements: Sequence[str],
    raw_limits: Dict[str, int],
    use_dbe_filter: bool,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    loss_matches = explain_neutral_losses(precursor_mz, fragments, ppm)
    fragment_library = build_fragment_formula_library(
        fragments=fragments,
        polarity=polarity,
        ppm=ppm,
        allowed_elements=allowed_elements,
        raw_limits=raw_limits,
    )

    for adduct in adducts:
        neutral_mass = precursor_mz - adduct.mass_shift
        if neutral_mass <= 0:
            continue
        limits = effective_limits(neutral_mass, allowed_elements, raw_limits)
        carbon_range = estimate_carbon_range(neutral_mass, fragments)
        matches = enumerate_formulas(
            target_mass=neutral_mass,
            ppm=ppm,
            allowed_elements=allowed_elements,
            limits=limits,
            use_dbe_filter=use_dbe_filter,
            carbon_range=carbon_range,
        )
        for counts, calc_mass, dbe in matches:
            calculated_mz = calc_mass + adduct.mass_shift
            ppm_error = (calculated_mz - precursor_mz) / precursor_mz * 1_000_000.0
            explained_fragments = count_explained_fragments(counts, fragment_library)
            ranking_score = abs(ppm_error) + hetero_penalty(counts) - 0.35 * explained_fragments
            elem_level = get_element_level(counts)

            candidates.append(
                Candidate(
                    adduct=adduct.name,
                    formula=format_formula(counts),
                    neutral_mass=calc_mass,
                    calculated_mz=calculated_mz,
                    ppm_error=ppm_error,
                    dbe=dbe,
                    counts=counts,
                    loss_matches=loss_matches,
                    explained_fragments=explained_fragments,
                    ranking_score=ranking_score,
                    element_level=elem_level
                )
            )

    # ====================== 你要的终极排序 ======================
    # 1. element_level: 0(CHO) → 1(CHON) → 2(OTHER)
    # 2. adduct priority: [M-H]- → [M+Cl]- → [M+FA-H]- → [M+Ac-H]-
    # 3. 质量准确度
    adduct_priority = {adduct.name: adduct.priority for adduct in adducts}
    candidates.sort(
        key=lambda item: (
            item.element_level,          # 先按类型：CHO全局先出
            adduct_priority[item.adduct],# 同类型内按加和顺序
            item.ranking_score,
            abs(item.ppm_error)
        )
    )
    return candidates
# =================================================================


def best_candidate_per_primary_adduct(candidates: Sequence[Candidate], polarity: str) -> Optional[Candidate]:
    primary = "[M+H]+" if polarity == "positive" else "[M-H]-"
    primary_hits = [item for item in candidates if item.adduct == primary]
    if primary_hits:
        return primary_hits[0]
    return candidates[0] if candidates else None


def print_summary(
    precursor_mz: float,
    polarity: str,
    fragments: Sequence[float],
    ppm: float,
    adducts: Sequence[Adduct],
    allowed_elements: List[str],
) -> None:
    print(f"Precursor m/z : {precursor_mz:.6f}")
    print(f"Polarity      : {polarity}")
    print(f"Tolerance     : {ppm:g} ppm")
    print(f"Adducts       : {', '.join(adduct.name for adduct in adducts)}")
    print(f"Elements      : {', '.join(allowed_elements)}")
    if fragments:
        print(f"Fragments     : {', '.join(f'{f:.4f}' for f in fragments)}")
    else:
        print("Fragments     : none")
    print()


def print_carbon_ranges(precursor_mz: float, fragments: Sequence[float], adducts: Sequence[Adduct]) -> None:
    print("Auto carbon range:")
    for adduct in adducts:
        nm = precursor_mz - adduct.mass_shift
        if nm <= 0:
            continue
        c_min, c_max = estimate_carbon_range(nm, fragments)
        print(f"- {adduct.name}: C{c_min}~C{c_max}")
    print()


def print_best_formula(candidates: Sequence[Candidate], polarity: str) -> None:
    best = best_candidate_per_primary_adduct(candidates, polarity)
    if not best:
        return
    print("Best formula (CHO priority):")
    print(f"{best.formula}  {best.adduct} | ppm {best.ppm_error:.3f} | DBE {best.dbe:.1f}\n")


def print_candidates(candidates: Sequence[Candidate], top_n: int) -> None:
    if not candidates:
        print("No candidates.")
        return

    header = f"{'Rank':>4}  {'Adduct':<12}  {'Formula':<18}  {'NeutralMass':>12}  {'Calc m/z':>12}  {'ppm':>9}  {'DBE':>6}  {'Type':<6}"
    print(header)
    print("-" * len(header))
    type_map = {0:"CHO",1:"CHON",2:"OTHER"}
    for i, item in enumerate(candidates[:top_n], 1):
        t = type_map[item.element_level]
        print(f"{i:>4}  {item.adduct:<12}  {item.formula:<18}  {item.neutral_mass:>12.6f}  {item.calculated_mz:>12.6f}  {item.ppm_error:>9.3f}  {item.dbe:>6.1f}  {t:<6}")

    print("\n✅ 顺序：所有加和CHO → 所有加和CHON → 所有加和OTHER")


def run_formula_prediction(
    precursor_mz: float,
    polarity: str,
    fragments: Sequence[float],
    ppm: float = 5.0,
    elements: object = "C,H,N,O,S,P",
    max_counts: object = None,
    adducts: object = None,
    top: int = 200,
    use_dbe_filter: bool = True,
) -> List[Candidate]:
    polarity = normalize_polarity(polarity)
    fragments = [float(x) for x in fragments]
    allowed_elements = normalize_elements_input(elements)
    raw_limits = normalize_max_counts_input(max_counts)
    adducts = normalize_adducts_input(polarity, adducts)

    print_summary(precursor_mz, polarity, fragments, ppm, adducts, allowed_elements)
    print_carbon_ranges(precursor_mz, fragments, adducts)
    candidates = build_candidates(
        precursor_mz=precursor_mz, polarity=polarity, adducts=adducts,
        fragments=fragments, ppm=ppm, allowed_elements=allowed_elements,
        raw_limits=raw_limits, use_dbe_filter=use_dbe_filter
    )
    print_best_formula(candidates, polarity)
    print_candidates(candidates, top)
    return candidates


def main() -> None:
    run_formula_prediction(
        precursor_mz=float(RUN_CONFIG["precursor_mz"]),
        polarity=str(RUN_CONFIG["polarity"]),
        fragments=RUN_CONFIG["fragments"],
        ppm=float(RUN_CONFIG["ppm"]),
        elements=RUN_CONFIG["elements"],
        max_counts=RUN_CONFIG["max_counts"],
        adducts=RUN_CONFIG["adducts"],
        top=int(RUN_CONFIG["top"]),
        use_dbe_filter=bool(RUN_CONFIG["use_dbe_filter"]),
    )


if __name__ == "__main__":
    main()