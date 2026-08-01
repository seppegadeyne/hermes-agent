"""Runtime tests for tool-call loop guardrails."""

import json
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import FinalSummaryResult
from agent.tool_guardrails import ToolGuardrailDecision
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=usage)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def _loop_cap_config(cap_key: str) -> dict:
    return {
        "tool_loop_guardrails": {
            "loop_caps": {
                cap_key: 1,
            }
        }
    }


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    assert agent._tool_guardrail_halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return None

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins.resolve_pre_tool_block",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value="plugin policy"),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)




def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text
    assert agent.client.chat.completions.create.call_count == 3

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


def test_web_search_cap_synthesizes_existing_results_without_more_tools():
    """A terminal research cap should preserve evidence in a final answer.

    The cap still blocks the extra search, but the model receives one tool-free
    call so users get a useful partial report instead of only guardrail text.
    """
    agent = _make_agent(
        "web_search",
        "write_file",
        max_iterations=10,
        config=_loop_cap_config("max_web_searches"),
    )
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "web_search",
                    json.dumps({"query": "first query"}),
                    "c-first",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "web_search",
                    json.dumps({"query": "second query"}),
                    "c-capped",
                ),
                _mock_tool_call(
                    "write_file",
                    json.dumps({"path": "/tmp/must-not-run", "content": "unsafe"}),
                    "c-skipped-write",
                ),
            ],
        ),
        _mock_response(
            content="Useful partial report from the available evidence.",
            finish_reason="stop",
            tool_calls=None,
        ),
    ]
    agent.client.chat.completions.create.side_effect = responses
    agent._disable_streaming = True
    deltas: list = []
    agent.stream_delta_callback = lambda delta: deltas.append(delta)

    existing_result = json.dumps(
        {"data": {"web": [{"title": "Useful source", "url": "https://example.com"}]}}
    )
    with (
        patch("run_agent.handle_function_call", return_value=existing_result) as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("research this topic thoroughly")

    assert dispatch.call_count == 1
    assert not any(
        call.args and call.args[0] == "write_file"
        for call in dispatch.call_args_list
    )
    assert agent.client.chat.completions.create.call_count == 3
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["code"] == "loop_web_search_cap"
    assert result["api_calls"] == 3
    assert agent.session_api_calls == 1
    assert result["final_response"] == "Useful partial report from the available evidence."
    matching_assistant_messages = [
        message
        for message in result["messages"]
        if message.get("role") == "assistant"
        and message.get("content") == result["final_response"]
    ]
    assert len(matching_assistant_messages) == 1
    assert matching_assistant_messages[0]["finish_reason"] == "stop"
    assert not any(
        message.get("role") == "user"
        and "tool-call guardrail stopped further research" in message.get("content", "")
        for message in result["messages"]
    )
    skipped_write = next(
        message
        for message in result["messages"]
        if message.get("tool_call_id") == "c-skipped-write"
    )
    assert "terminal tool-loop cap" in skipped_write["content"]

    summary_request = agent.client.chat.completions.create.call_args_list[-1].kwargs
    assert "tools" not in summary_request
    assert "tool_choice" not in summary_request
    assert any(
        message.get("role") == "tool" and "Useful source" in message.get("content", "")
        for message in summary_request["messages"]
    )
    assert summary_request["messages"][-1]["role"] == "user"
    assert "Do not call any more tools" in summary_request["messages"][-1]["content"]

    text_deltas = [delta for delta in deltas if isinstance(delta, str)]
    assert result["final_response"] in text_deltas


def test_subagent_cap_also_requests_tool_free_final_response():
    agent = _make_agent("delegate_task")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Partial report from completed subagents.",
        finish_reason="stop",
        tool_calls=None,
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_subagent_cap",
        message="cap reached",
        tool_name="delegate_task",
        count=50,
    )

    response = agent._toolguard_final_response(
        [{"role": "user", "content": "research this"}],
        decision,
        api_call_count=2,
    )

    assert response.text == "Partial report from completed subagents."
    assert response.api_attempts == 1
    assert response.already_streamed is False
    summary_request = agent.client.chat.completions.create.call_args.kwargs
    assert "tools" not in summary_request
    assert "tool_choice" not in summary_request


def test_subagent_batch_that_would_exceed_cap_is_blocked_before_dispatch():
    agent = _make_agent(
        "delegate_task",
        config=_loop_cap_config("max_subagents"),
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "delegate_task",
                    json.dumps(
                        {
                            "tasks": [
                                {"goal": "first child"},
                                {"goal": "second child"},
                            ]
                        }
                    ),
                    call_id="call-batched-delegation",
                )
            ],
        ),
        _mock_response(content="No children were started; reporting existing work."),
    ]

    with (
        patch.object(agent, "_dispatch_delegate_task") as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("delegate this research")

    dispatch.assert_not_called()
    assert result["guardrail"]["code"] == "loop_subagent_cap"
    assert result["api_calls"] == 2
    assert result["final_response"] == (
        "No children were started; reporting existing work."
    )


def test_terminal_loop_cap_summary_failure_uses_controlled_halt_fallback():
    agent = _make_agent("web_search")
    agent.client.chat.completions.create.side_effect = RuntimeError("provider unavailable")
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )
    messages = [
        {"role": "user", "content": "research this"},
        {"role": "tool", "content": "useful existing result"},
    ]

    response = agent._toolguard_final_response(messages, decision, api_call_count=2)

    assert "stopped retrying web_search" in response.text
    assert "loop_web_search_cap" in response.text
    assert response.api_attempts == 1
    assert agent.client.chat.completions.create.call_count == 1


def test_terminal_loop_cap_empty_summary_is_not_retried():
    agent = _make_agent("web_search")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="",
        finish_reason="stop",
        tool_calls=None,
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )

    response = agent._toolguard_final_response(
        [{"role": "user", "content": "research this"}],
        decision,
        api_call_count=2,
    )

    assert "loop_web_search_cap" in response.text
    assert response.api_attempts == 1
    assert agent.client.chat.completions.create.call_count == 1


def test_terminal_loop_cap_summary_records_usage_and_redacts_persisted_content():
    agent = _make_agent("web_search")
    exposed_secret = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
    agent.client.chat.completions.create.return_value = _mock_response(
        content=f"Summary accidentally included {exposed_secret}",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )
    messages = [{"role": "user", "content": "research this"}]

    with patch.dict(os.environ, {"HERMES_REDACT_SECRETS": "true"}):
        response = agent._toolguard_final_response(
            messages,
            decision,
            api_call_count=2,
        )

    assert exposed_secret not in response.text
    assert exposed_secret not in messages[-1]["content"]
    assert agent.session_api_calls == 1
    assert agent.session_prompt_tokens == 11
    assert agent.session_completion_tokens == 7
    assert agent.session_total_tokens == 18


def test_iteration_limit_summary_attempt_is_in_returned_api_call_count():
    agent = _make_agent("web_search", max_iterations=1)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "web_search",
                    json.dumps({"query": "one query"}),
                    call_id="call-one",
                )
            ],
        ),
        _mock_response(content="Iteration-limit partial report."),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="source evidence"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("research this")

    assert result["api_calls"] == 2
    assert result["final_response"] == "Iteration-limit partial report."


def test_terminal_loop_cap_codex_summary_is_tool_free_and_buffered():
    agent = _make_agent("web_search")
    agent.api_mode = "codex_responses"
    deltas = []
    agent.stream_delta_callback = deltas.append
    build_calls = []

    def build_kwargs(api_messages, tools_for_api=None):
        build_calls.append((api_messages, tools_for_api))
        return {"model": "codex-test", "input": api_messages}

    def run_codex_stream(_request):
        return object()

    transport = SimpleNamespace(
        preflight_kwargs=lambda request, **_kwargs: request,
        normalize_response=lambda _response: SimpleNamespace(
            content="Codex cap summary",
            tool_calls=None,
            reasoning=None,
            finish_reason="stop",
        )
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )

    with (
        patch.object(agent, "_build_api_kwargs", side_effect=build_kwargs),
        patch.object(agent, "_run_codex_stream", side_effect=run_codex_stream),
        patch.object(agent, "_get_transport", return_value=transport),
        patch(
            "agent.relay_llm.execute_current",
            side_effect=AssertionError("Codex must not nest non-streaming Relay"),
        ),
    ):
        response = agent._toolguard_final_response(
            [{"role": "user", "content": "research this"}],
            decision,
            api_call_count=2,
        )

    assert response == FinalSummaryResult("Codex cap summary", 1, False)
    assert len(build_calls) == 1
    assert build_calls[0][1] == []
    assert deltas == []


def test_terminal_loop_cap_bedrock_summary_uses_converse_transport_without_tools():
    agent = _make_agent("web_search")
    agent.api_mode = "bedrock_converse"
    build_calls = []

    def build_kwargs(api_messages, tools_for_api=None):
        build_calls.append((api_messages, tools_for_api))
        return {
            "__bedrock_converse__": True,
            "model": "bedrock-test",
            "messages": api_messages,
        }

    transport = SimpleNamespace(
        normalize_response=lambda _response: SimpleNamespace(
            content="Bedrock cap summary",
            tool_calls=None,
            reasoning=None,
            finish_reason="stop",
        )
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )

    with (
        patch.object(agent, "_build_api_kwargs", side_effect=build_kwargs),
        patch.object(agent, "_get_transport", return_value=transport),
        patch.object(agent, "_interruptible_api_call", return_value=object()) as dispatch,
    ):
        response = agent._toolguard_final_response(
            [{"role": "user", "content": "research this"}],
            decision,
            api_call_count=2,
        )

    assert response == FinalSummaryResult("Bedrock cap summary", 1, False)
    assert len(build_calls) == 1
    assert build_calls[0][1] == []
    dispatch.assert_called_once()


def test_terminal_loop_cap_anthropic_summary_is_tool_free():
    agent = _make_agent("web_search")
    agent.api_mode = "anthropic_messages"
    build_calls = []

    def build_kwargs(**kwargs):
        build_calls.append(kwargs)
        return {"model": kwargs["model"], "messages": kwargs["messages"]}

    transport = SimpleNamespace(
        normalize_response=lambda _response, **_kwargs: SimpleNamespace(
            content="Anthropic cap summary",
            tool_calls=None,
            reasoning=None,
            finish_reason="stop",
        ),
    )
    decision = ToolGuardrailDecision(
        action="block",
        code="loop_web_search_cap",
        message="cap reached",
        tool_name="web_search",
        count=50,
    )

    with (
        patch.object(
            agent,
            "_build_api_kwargs",
            side_effect=lambda messages, tools_for_api=None: build_kwargs(
                model="test",
                messages=messages,
                tools=tools_for_api,
            ),
        ),
        patch.object(agent, "_get_transport", return_value=transport),
        patch.object(agent, "_interruptible_api_call", return_value=object()),
    ):
        response = agent._toolguard_final_response(
            [{"role": "user", "content": "research this"}],
            decision,
            api_call_count=2,
        )

    assert response == FinalSummaryResult("Anthropic cap summary", 1, False)
    assert len(build_calls) == 1
    assert build_calls[0]["tools"] == []


def test_already_streamed_cap_summary_is_not_replayed_to_client():
    agent = _make_agent(
        "web_search",
        config=_loop_cap_config("max_web_searches"),
    )
    deltas = []
    agent.stream_delta_callback = deltas.append
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "web_search",
                    json.dumps({"query": "first query"}),
                    call_id="call-first",
                )
            ],
        ),
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    "web_search",
                    json.dumps({"query": "second query"}),
                    call_id="call-second",
                )
            ],
        ),
    ]

    def already_streamed(*_args, **_kwargs):
        agent.stream_delta_callback("Already streamed cap summary")
        return FinalSummaryResult("Already streamed cap summary", 1, True)

    with (
        patch("run_agent.handle_function_call", return_value="source evidence"),
        patch.object(
            agent,
            "_toolguard_final_response",
            side_effect=already_streamed,
        ),
    ):
        result = agent.run_conversation("research this")

    text_deltas = [delta for delta in deltas if isinstance(delta, str)]
    assert text_deltas.count("Already streamed cap summary") == 1
    assert result["api_calls"] == 3
