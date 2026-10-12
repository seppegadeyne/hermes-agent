"""Operator opt-in for per-task delegate_task pins (delegation.allow_model_selection).

Naming and semantics follow #85228: selection is off by default, values are exact
allowlists, and while it is off the fields are not advertised and every pin is refused,
so the default contract (children inherit, config pins all) is unchanged.
"""
import json
from unittest.mock import patch

import pytest

from tools.delegate_tool import _build_dynamic_schema_overrides, delegate_task
from tools.delegate_tool_model_selection import check_task_selection, model_selection_allowlist
from tests.tools.test_delegate_task_provider_pin import (
    PARENT_CREDS, PIN_CREDS, _fake_resolve, _make_mock_parent, _ok_child,
)

OTHER_ROUTE = {"provider": "lmstudio-x121", "model": "ling-3.0-tiny"}
ENABLED = {"allow_model_selection": True, "allowed_models": ["cheap-tier"]}


def _resolve(cfg, parent_agent):
    # Like the real resolver, an explicit model in the route config wins over the provider default.
    return dict(_fake_resolve(cfg, parent_agent), **({"model": cfg["model"]} if cfg.get("model") else {}))


def _delegate(tasks, cfg):
    with (
        patch("tools.delegate_tool._load_config", return_value=cfg),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_resolve),
        patch("run_agent.AIAgent") as MockAgent,
    ):
        MockAgent.side_effect = lambda *a, **k: _ok_child()
        out = json.loads(delegate_task(tasks=tasks, parent_agent=_make_mock_parent()))
    return out, [c.kwargs for c in MockAgent.call_args_list]


def _task_item_props(cfg):
    with patch("tools.delegate_tool._load_config", return_value=cfg):
        params = _build_dynamic_schema_overrides()["parameters"]
    return params["properties"]["tasks"]["items"]["properties"]


@pytest.mark.parametrize("cfg", [
    {},
    {"allowed_models": ["cheap-tier"], "allowed_routes": [OTHER_ROUTE]},  # lists without the opt-in
    {"allow_model_selection": True},  # opt-in without a list
    {"allow_model_selection": True, "allowed_models": "cheap-tier"},  # malformed list
])
@pytest.mark.parametrize("pin", [{"model": "cheap-tier"}, OTHER_ROUTE, {"provider": "lmstudio-x121"}])
def test_pins_refused_and_hidden_unless_enabled_with_an_allowlist(cfg, pin):
    out, built = _delegate([{"goal": "Extract every link from the sitemap", **pin}], cfg)
    assert out["error"].startswith("Task 0: per-task model selection is disabled") and built == []
    props = _task_item_props(cfg)
    assert "model" not in props and "provider" not in props


def test_unlisted_model_refuses_whole_batch_before_any_child():
    tasks = [
        {"goal": "Extract every link from the sitemap", "model": "cheap-tier"},
        {"goal": "Judge which links matter most", "model": "flagship-model"},
    ]
    out, built = _delegate(tasks, ENABLED)
    assert out["error"] == "Task 1: model 'flagship-model' is not allowed"
    assert built == []


def test_allowed_model_stays_on_batch_provider():
    tasks = [
        {"goal": "Extract every link from the sitemap", "model": "cheap-tier"},
        {"goal": "Judge which links matter most"},
    ]
    out, built = _delegate(tasks, ENABLED)
    assert "error" not in out
    assert [k["model"] for k in built] == ["cheap-tier", PARENT_CREDS["model"]]
    assert {k["provider"] for k in built} == {PARENT_CREDS["provider"]}


@pytest.mark.parametrize("pin, message", [
    ({"provider": "lmstudio-x121", "model": "cheap-tier"}, "route 'lmstudio-x121:cheap-tier' is not allowed"),
    ({"provider": "lmstudio-x121"}, "provider 'lmstudio-x121' needs a model from delegation.allowed_routes"),
])
def test_allowed_model_is_not_allowed_on_another_provider(pin, message):
    out, built = _delegate([{"goal": "Extract links", **pin}], {**ENABLED, "allowed_routes": [OTHER_ROUTE]})
    assert out["error"] == f"Task 0: {message}" and built == []


def test_other_provider_needs_its_exact_route():
    cfg = {"allow_model_selection": True, "allowed_routes": [OTHER_ROUTE]}
    out, built = _delegate([{"goal": "Extract every link from the sitemap", **OTHER_ROUTE}], cfg)
    assert "error" not in out
    assert (built[0]["provider"], built[0]["model"]) == (PIN_CREDS["provider"], PIN_CREDS["model"])
    # The route's model alone is not a batch-provider model.
    out, built = _delegate([{"goal": "Extract links", "model": "ling-3.0-tiny"}], cfg)
    assert out["error"] == "Task 0: model 'ling-3.0-tiny' is not allowed" and built == []


def test_schema_advertises_only_allowed_values():
    props = _task_item_props({**ENABLED, "allowed_routes": [OTHER_ROUTE]})
    assert props["model"]["enum"] == ["cheap-tier", "ling-3.0-tiny"]
    assert props["provider"]["enum"] == ["lmstudio-x121"]
    assert "goal" in props
    props = _task_item_props(ENABLED)
    assert props["model"]["enum"] == ["cheap-tier"] and "provider" not in props


def test_malformed_allowlist_entries_are_ignored():
    cfg = {
        "allow_model_selection": "true",
        "allowed_models": ["haiku", " ", 3, "haiku"],
        "allowed_routes": ["openrouter:x", {"provider": "openrouter"}, {"model": "x"}, OTHER_ROUTE, OTHER_ROUTE],
    }
    assert model_selection_allowlist(cfg) == (["haiku"], [("lmstudio-x121", "ling-3.0-tiny")])
    assert check_task_selection({"goal": "x"}, [], []) is None
