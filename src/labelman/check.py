"""Check (validate) a labelman.yaml file."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import CategoryMode, ParseError, TermList, parse


@dataclass
class CheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    num_categories: int
    num_terms: int
    num_global_terms: int = 0


def check(source: str) -> CheckResult:
    """Validate a labelman.yaml file for structural correctness and semantic consistency.

    Returns a CheckResult with errors, warnings, and summary counts.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        term_list = parse(source)
    except ParseError as e:
        return CheckResult(
            ok=False,
            errors=e.errors,
            warnings=[],
            num_categories=0,
            num_terms=0,
        )

    total_terms = 0

    for cat in term_list.categories:
        num_terms = len(cat.terms)
        total_terms += num_terms

        if cat.mode == CategoryMode.EXACTLY_ONE and num_terms < 2:
            warnings.append(
                f"Category '{cat.name}': exactly-one mode with only {num_terms} term(s) "
                f"is likely a mistake"
            )

        # Degenerate thresholds
        if cat.threshold is not None:
            if cat.threshold == 0.0:
                warnings.append(
                    f"Category '{cat.name}': threshold of 0.0 will match everything"
                )
            elif cat.threshold == 1.0:
                warnings.append(
                    f"Category '{cat.name}': threshold of 1.0 will match almost nothing"
                )

        for term in cat.terms:
            if term.threshold is not None:
                if term.threshold == 0.0:
                    warnings.append(
                        f"Term '{cat.name}/{term.term}': threshold of 0.0 will match everything"
                    )
                elif term.threshold == 1.0:
                    warnings.append(
                        f"Term '{cat.name}/{term.term}': threshold of 1.0 will match almost nothing"
                    )

    if term_list.defaults.threshold == 0.0:
        warnings.append("Global default threshold of 0.0 will match everything")
    elif term_list.defaults.threshold == 1.0:
        warnings.append("Global default threshold of 1.0 will match almost nothing")

    return CheckResult(
        ok=True,
        errors=[],
        warnings=warnings,
        num_categories=len(term_list.categories),
        num_terms=total_terms,
        num_global_terms=len(term_list.global_terms),
    )
