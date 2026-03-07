"""Schema definitions and parsing for labelman.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class CategoryMode(Enum):
    EXACTLY_ONE = "exactly-one"
    ZERO_OR_ONE = "zero-or-one"
    ZERO_OR_MORE = "zero-or-more"


@dataclass
class Term:
    term: str
    threshold: Optional[float] = None


@dataclass
class Category:
    name: str
    mode: CategoryMode
    terms: list[Term]
    threshold: Optional[float] = None


@dataclass
class Defaults:
    threshold: float


@dataclass
class IntegrationConfig:
    """Configuration for a single integration (blip or clip).

    Either endpoint or script must be set. If script is set, it is used
    directly and endpoint is ignored. If only endpoint is set, the
    built-in script is used with that endpoint.
    """
    endpoint: Optional[str] = None
    script: Optional[str] = None


@dataclass
class Integrations:
    blip: Optional[IntegrationConfig] = None
    clip: Optional[IntegrationConfig] = None


@dataclass
class TermList:
    defaults: Defaults
    categories: list[Category]
    integrations: Integrations
    global_terms: list[str]

    def effective_threshold(self, category: Category, term: Term) -> float:
        if term.threshold is not None:
            return term.threshold
        if category.threshold is not None:
            return category.threshold
        return self.defaults.threshold


class ParseError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


_VALID_MODES = {m.value for m in CategoryMode}


def _validate_threshold(value, context: str) -> list[str]:
    errors = []
    if not isinstance(value, (int, float)):
        errors.append(f"{context}: threshold must be a number, got {type(value).__name__}")
    elif not (0.0 <= value <= 1.0):
        errors.append(f"{context}: threshold must be between 0.0 and 1.0, got {value}")
    return errors


def _parse_integration(raw: dict, name: str, errors: list[str]) -> Optional[IntegrationConfig]:
    if not isinstance(raw, dict):
        errors.append(f"integrations.{name}: must be a mapping")
        return None
    endpoint = raw.get("endpoint")
    script = raw.get("script")
    if endpoint is not None and not isinstance(endpoint, str):
        errors.append(f"integrations.{name}.endpoint: must be a string")
        endpoint = None
    if script is not None and not isinstance(script, str):
        errors.append(f"integrations.{name}.script: must be a string")
        script = None
    if endpoint is None and script is None:
        errors.append(f"integrations.{name}: must specify 'endpoint' or 'script'")
        return None
    return IntegrationConfig(endpoint=endpoint, script=script)


def _load_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text()
    if isinstance(source, str) and "\n" not in source and len(source) < 4096:
        p = Path(source)
        if p.is_file():
            return p.read_text()
    return source


def parse(source: str | Path) -> TermList:
    """Parse a labelman.yaml file or string into a TermList.

    Args:
        source: Either a file path (str or Path) or a YAML string.

    Raises:
        ParseError: If the file has structural or semantic errors.
    """
    text = _load_text(source)
    errors: list[str] = []

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ParseError([f"Invalid YAML: {e}"])

    if not isinstance(data, dict):
        raise ParseError(["Top level must be a YAML mapping"])

    # --- defaults ---
    if "defaults" not in data:
        errors.append("Missing required key: 'defaults'")
        defaults_obj = None
    else:
        defaults_raw = data["defaults"]
        if not isinstance(defaults_raw, dict):
            errors.append("'defaults' must be a mapping")
            defaults_obj = None
        elif "threshold" not in defaults_raw:
            errors.append("'defaults' missing required key: 'threshold'")
            defaults_obj = None
        else:
            t_errors = _validate_threshold(defaults_raw["threshold"], "defaults.threshold")
            errors.extend(t_errors)
            defaults_obj = Defaults(threshold=float(defaults_raw["threshold"])) if not t_errors else None

    # --- integrations (optional) ---
    integrations_obj = Integrations()
    if "integrations" in data:
        integ_raw = data["integrations"]
        if not isinstance(integ_raw, dict):
            errors.append("'integrations' must be a mapping")
        else:
            if "blip" in integ_raw:
                integrations_obj.blip = _parse_integration(integ_raw["blip"], "blip", errors)
            if "clip" in integ_raw:
                integrations_obj.clip = _parse_integration(integ_raw["clip"], "clip", errors)

    # --- global_terms (optional) ---
    global_terms_list: list[str] = []
    if "global_terms" in data:
        gt_raw = data["global_terms"]
        if not isinstance(gt_raw, list):
            errors.append("'global_terms' must be a list")
        else:
            for i, item in enumerate(gt_raw):
                if not isinstance(item, str):
                    errors.append(f"global_terms[{i}]: must be a string, got {type(item).__name__}")
                elif not item.strip():
                    errors.append(f"global_terms[{i}]: must not be empty")
                else:
                    global_terms_list.append(item.strip())

    # --- categories ---
    if "categories" not in data:
        errors.append("Missing required key: 'categories'")
        categories_list = []
    else:
        cats_raw = data["categories"]
        if not isinstance(cats_raw, list):
            errors.append("'categories' must be a list")
            categories_list = []
        elif len(cats_raw) == 0:
            errors.append("'categories' must not be empty")
            categories_list = []
        else:
            categories_list = []
            seen_cat_names: set[str] = set()

            for i, cat_raw in enumerate(cats_raw):
                cat_ctx = f"categories[{i}]"

                if not isinstance(cat_raw, dict):
                    errors.append(f"{cat_ctx}: must be a mapping")
                    continue

                # name
                if "name" not in cat_raw:
                    errors.append(f"{cat_ctx}: missing required key 'name'")
                    cat_name = None
                else:
                    cat_name = str(cat_raw["name"])
                    if cat_name in seen_cat_names:
                        errors.append(f"{cat_ctx}: duplicate category name '{cat_name}'")
                    seen_cat_names.add(cat_name)

                # mode
                if "mode" not in cat_raw:
                    errors.append(f"{cat_ctx}: missing required key 'mode'")
                    cat_mode = None
                else:
                    mode_val = cat_raw["mode"]
                    if mode_val not in _VALID_MODES:
                        errors.append(
                            f"{cat_ctx}: invalid mode '{mode_val}', "
                            f"must be one of: {', '.join(sorted(_VALID_MODES))}"
                        )
                        cat_mode = None
                    else:
                        cat_mode = CategoryMode(mode_val)

                # category threshold
                cat_threshold = None
                if "threshold" in cat_raw:
                    t_errors = _validate_threshold(cat_raw["threshold"], f"{cat_ctx}.threshold")
                    errors.extend(t_errors)
                    if not t_errors:
                        cat_threshold = float(cat_raw["threshold"])

                # terms
                if "terms" not in cat_raw:
                    errors.append(f"{cat_ctx}: missing required key 'terms'")
                    terms_list = []
                else:
                    terms_raw = cat_raw["terms"]
                    if not isinstance(terms_raw, list):
                        errors.append(f"{cat_ctx}.terms: must be a list")
                        terms_list = []
                    elif len(terms_raw) == 0:
                        errors.append(f"{cat_ctx}.terms: must not be empty")
                        terms_list = []
                    else:
                        terms_list = []
                        seen_term_names: set[str] = set()

                        for j, term_raw in enumerate(terms_raw):
                            term_ctx = f"{cat_ctx}.terms[{j}]"

                            if isinstance(term_raw, str):
                                term_name = term_raw
                                term_threshold = None
                            elif isinstance(term_raw, dict):
                                if "term" not in term_raw:
                                    errors.append(f"{term_ctx}: missing required key 'term'")
                                    continue
                                term_name = str(term_raw["term"])
                                if not term_name:
                                    errors.append(f"{term_ctx}: 'term' must not be empty")
                                    continue

                                term_threshold = None
                                if "threshold" in term_raw:
                                    t_errors = _validate_threshold(
                                        term_raw["threshold"], f"{term_ctx}.threshold"
                                    )
                                    errors.extend(t_errors)
                                    if not t_errors:
                                        term_threshold = float(term_raw["threshold"])
                            else:
                                errors.append(f"{term_ctx}: must be a string or mapping")
                                continue

                            if term_name in seen_term_names:
                                errors.append(
                                    f"{term_ctx}: duplicate term '{term_name}' in category "
                                    f"'{cat_name or cat_ctx}'"
                                )
                            seen_term_names.add(term_name)
                            terms_list.append(Term(term=term_name, threshold=term_threshold))

                if cat_name is not None and cat_mode is not None:
                    categories_list.append(
                        Category(
                            name=cat_name,
                            mode=cat_mode,
                            terms=terms_list,
                            threshold=cat_threshold,
                        )
                    )

    if errors:
        raise ParseError(errors)

    assert defaults_obj is not None
    return TermList(
        defaults=defaults_obj,
        categories=categories_list,
        integrations=integrations_obj,
        global_terms=global_terms_list,
    )
