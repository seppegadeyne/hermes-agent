"""Operator opt-in for per-task ``model``/``provider`` pins on ``delegate_task``.

Same contract as the opt-in model selection proposed in #85228: the model never gains
authority over compute the operator did not hand it. Pins stay off (and the fields are not
advertised) until ``delegation.allow_model_selection`` is true and an exact allowlist is set:

- ``allowed_models``: exact model names a task may pin on the batch/parent provider and its
  credentials (e.g. ``haiku`` under an Opus parent on a subscription provider);
- ``allowed_routes``: exact ``{provider, model}`` pairs a task may pin on another provider,
  resolved like ``delegation.provider`` (full credential bundle).

The schema only advertises listed values (as enums), and any pin outside the lists fails
the whole call before a route resolves or a child is built.
"""
from __future__ import annotations

from typing import Any, Optional

from utils import is_truthy_value

_PIN_FIELDS = ("provider", "model")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    items = (_text(item) for item in value) if isinstance(value, (list, tuple)) else ()
    return list(dict.fromkeys(item for item in items if item))


def model_selection_allowlist(cfg: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """``(allowed_models, allowed_routes)`` when selection is enabled, else two empty lists.
    Malformed entries are dropped; a route needs both a provider and a model."""
    if not isinstance(cfg, dict) or not is_truthy_value(cfg.get("allow_model_selection"), default=False):
        return [], []
    raw_routes = cfg.get("allowed_routes")
    routes: list[tuple[str, str]] = []
    for entry in raw_routes if isinstance(raw_routes, (list, tuple)) else ():
        route = (_text(entry.get("provider")), _text(entry.get("model"))) if isinstance(entry, dict) else ("", "")
        if all(route) and route not in routes:
            routes.append(route)
    return _string_list(cfg.get("allowed_models")), routes


def check_task_selection(task: dict[str, Any], models: list[str], routes: list[tuple[str, str]]) -> Optional[str]:
    """Refusal message for a task pin outside the allowlist, else None (also for unpinned tasks)."""
    provider, model = _text(task.get("provider")), _text(task.get("model"))
    if not provider and not model:
        return None
    if not models and not routes:
        return ("per-task model selection is disabled (operator setting: "
                "delegation.allow_model_selection + delegation.allowed_models)")
    if not provider:
        return None if model in models else f"model '{model}' is not allowed"
    if (provider, model) in routes:
        return None
    if not model:
        return f"provider '{provider}' needs a model from delegation.allowed_routes"
    return f"route '{provider}:{model}' is not allowed"


def refuse_unselected_task_pins(task_list: list[dict[str, Any]], cfg: Any) -> Optional[str]:
    """First refusal across the batch (``Task <i>: ...``), or None when every pin is allowed."""
    models, routes = model_selection_allowlist(cfg)
    # Resolved through the facade so tests that cover pin resolution can open the gate.
    from tools import delegate_tool
    for i, task in enumerate(task_list):
        refusal = delegate_tool.check_task_selection(task, models, routes)
        if refusal:
            return f"Task {i}: {refusal}"
    return None


def model_selection_schema(item_props: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """Task-item properties with the pin fields constrained to the allowlist, or removed when it is empty."""
    models, routes = model_selection_allowlist(cfg)
    props = {k: v for k, v in item_props.items() if k not in _PIN_FIELDS}
    model_enum = list(dict.fromkeys([*models, *(m for _, m in routes)]))
    if not model_enum:
        return props
    pairs = ", ".join(f"{p}:{m}" for p, m in routes)
    props["model"] = {**item_props["model"], "enum": model_enum, "description": (
        "Operator-approved model for THIS child only. Without provider it runs on the batch/parent "
        f"provider (allowed: {', '.join(models) or 'none'})"
        + (f"; other providers only as these exact provider:model pairs: {pairs}" if routes else "")
        + ". Use a cheaper model for mechanical work (extraction, checklists); omit to inherit."
    )}
    if routes:
        props["provider"] = {**item_props["provider"], "enum": list(dict.fromkeys(p for p, _ in routes)),
                             "description": ("Operator-approved provider for THIS child only; valid only "
                                             f"together with model as one of: {pairs}. Omit to stay on the "
                                             "batch/parent provider.")}
    return props
