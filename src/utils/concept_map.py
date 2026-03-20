"""Bidirectional mapping between CamelCase and spaced IFRS concept names.

Heuristic-only version — no tags.jsonl dependency.
"""

import re


def camel_to_words(name: str) -> str:
    """CamelCase -> lowercase words.
    e.g. 'AdjustmentsForAmortisationExpense' -> 'adjustments for amortisation expense'
    """
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    words = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", words)
    return words.lower()


def words_to_camel(words: str) -> str:
    """Lowercase words -> CamelCase.
    e.g. 'rental income' -> 'RentalIncome'
    """
    return "".join(w.capitalize() for w in words.split())


def concept_to_spaced(concept: str) -> str:
    """'ifrs-full:RentalIncome' -> 'ifrs-full: rental income'"""
    if ":" not in concept:
        return concept
    prefix, name = concept.split(":", 1)
    return f"{prefix}: {camel_to_words(name)}"


def spaced_to_concept(spaced: str) -> str:
    """'ifrs-full: rental income' -> 'ifrs-full:RentalIncome' (heuristic)."""
    if ": " not in spaced:
        return spaced
    prefix, name = spaced.split(": ", 1)
    return f"{prefix}:{words_to_camel(name)}"


def map_back_to_camel(spaced: str) -> str:
    """Reverse-map a spaced concept to CamelCase (heuristic only)."""
    return spaced_to_concept(spaced)
