"""Exception classes for user-facing labelman errors.

Anything that inherits from LabelmanError is expected to be caught at the
CLI entry point and printed as a clean message (no traceback). Uncaught
exceptions of other types are real bugs.
"""

from __future__ import annotations


class LabelmanError(Exception):
    """Base class for user-facing errors surfaced by the CLI."""


class IntegrationError(LabelmanError):
    """A call to an external integration (BLIP/CLIP/LLM/Qwen-VL) failed."""


class UIServerError(LabelmanError):
    """The labeling UI server could not start or bind."""
