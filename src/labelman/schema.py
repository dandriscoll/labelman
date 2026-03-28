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
    ask: Optional[str] = None
    ask_negative: Optional[str] = None


@dataclass
class Category:
    name: str
    mode: CategoryMode
    terms: list[Term]
    threshold: Optional[float] = None
    question: Optional[str] = None
    ask: Optional[str] = None
    question_answer_max_tokens: Optional[int] = None


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
class LLMConfig:
    """Configuration for an LLM integration (litellm-compatible).

    Used to refine BLIP VQA answers into category labels.
    """
    endpoint: str
    model: Optional[str] = None


@dataclass
class Integrations:
    blip: Optional[IntegrationConfig] = None
    clip: Optional[IntegrationConfig] = None
    llm: Optional[LLMConfig] = None


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


def _parse_llm_integration(raw: dict, errors: list[str]) -> Optional[LLMConfig]:
    if not isinstance(raw, dict):
        errors.append("integrations.llm: must be a mapping")
        return None
    endpoint = raw.get("endpoint")
    if endpoint is None or not isinstance(endpoint, str):
        errors.append("integrations.llm: must specify 'endpoint' as a string")
        return None
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        errors.append("integrations.llm.model: must be a string")
        model = None
    return LLMConfig(endpoint=endpoint, model=model)


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
            if "llm" in integ_raw:
                integrations_obj.llm = _parse_llm_integration(integ_raw["llm"], errors)

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

                # question (optional)
                cat_question = None
                if "question" in cat_raw:
                    q_val = cat_raw["question"]
                    if not isinstance(q_val, str):
                        errors.append(f"{cat_ctx}.question: must be a string")
                    elif not q_val.strip():
                        errors.append(f"{cat_ctx}.question: must not be empty")
                    else:
                        cat_question = q_val.strip()

                # ask (optional — sent to BLIP instead of question)
                cat_ask = None
                if "ask" in cat_raw:
                    p_val = cat_raw["ask"]
                    if not isinstance(p_val, str):
                        errors.append(f"{cat_ctx}.ask: must be a string")
                    elif not p_val.strip():
                        errors.append(f"{cat_ctx}.ask: must not be empty")
                    else:
                        cat_ask = p_val.strip()

                # question_answer_max_tokens (optional, only meaningful with question)
                cat_max_tokens = None
                if "question_answer_max_tokens" in cat_raw:
                    mt_val = cat_raw["question_answer_max_tokens"]
                    if not isinstance(mt_val, int) or isinstance(mt_val, bool):
                        errors.append(f"{cat_ctx}.question_answer_max_tokens: must be a positive integer")
                    elif mt_val < 1:
                        errors.append(f"{cat_ctx}.question_answer_max_tokens: must be a positive integer, got {mt_val}")
                    else:
                        cat_max_tokens = mt_val

                # terms (optional when question is set)
                has_question = cat_question is not None
                if "terms" not in cat_raw:
                    if not has_question:
                        errors.append(f"{cat_ctx}: missing required key 'terms' (or provide a 'question')")
                    terms_list = []
                else:
                    terms_raw = cat_raw["terms"]
                    if not isinstance(terms_raw, list):
                        errors.append(f"{cat_ctx}.terms: must be a list")
                        terms_list = []
                    elif len(terms_raw) == 0:
                        if not has_question:
                            errors.append(f"{cat_ctx}.terms: must not be empty (or provide a 'question')")
                        terms_list = []
                    else:
                        terms_list = []
                        seen_term_names: set[str] = set()

                        for j, term_raw in enumerate(terms_raw):
                            term_ctx = f"{cat_ctx}.terms[{j}]"

                            if isinstance(term_raw, str):
                                term_name = term_raw
                                term_threshold = None
                                term_ask = None
                                term_ask_negative = None
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

                                term_ask = None
                                if "ask" in term_raw:
                                    tp_val = term_raw["ask"]
                                    if not isinstance(tp_val, str):
                                        errors.append(f"{term_ctx}.ask: must be a string")
                                    elif not tp_val.strip():
                                        errors.append(f"{term_ctx}.ask: must not be empty")
                                    else:
                                        term_ask = tp_val.strip()

                                term_ask_negative = None
                                if "ask_negative" in term_raw:
                                    tn_val = term_raw["ask_negative"]
                                    if not isinstance(tn_val, str):
                                        errors.append(f"{term_ctx}.ask_negative: must be a string")
                                    elif not tn_val.strip():
                                        errors.append(f"{term_ctx}.ask_negative: must not be empty")
                                    else:
                                        term_ask_negative = tn_val.strip()
                            else:
                                errors.append(f"{term_ctx}: must be a string or mapping")
                                continue

                            if term_name in seen_term_names:
                                errors.append(
                                    f"{term_ctx}: duplicate term '{term_name}' in category "
                                    f"'{cat_name or cat_ctx}'"
                                )
                            seen_term_names.add(term_name)
                            terms_list.append(Term(term=term_name, threshold=term_threshold,
                                                   ask=term_ask,
                                                   ask_negative=term_ask_negative))

                if cat_name is not None and cat_mode is not None:
                    categories_list.append(
                        Category(
                            name=cat_name,
                            mode=cat_mode,
                            terms=terms_list,
                            threshold=cat_threshold,
                            question=cat_question,
                            ask=cat_ask,
                            question_answer_max_tokens=cat_max_tokens,
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
