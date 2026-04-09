"""End-to-end tests for the tool_efficiency evaluator."""

import json
import subprocess
import sys

import pytest

EVALUATOR = "evaluators/tool_efficiency/tool_efficiency.py"


def _run(payload: dict) -> dict:
    result = subprocess.run(
        [sys.executable, EVALUATOR],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return json.loads(result.stdout)


def _make_input(invocations, config=None):
    return {
        "protocol_version": "1.0",
        "metric_name": "tool_efficiency",
        "threshold": 0.5,
        "config": config or {},
        "invocations": invocations,
    }


def _inv(inv_id, tool_calls=None, tool_responses=None):
    inv = {"invocation_id": inv_id}
    if tool_calls is not None or tool_responses is not None:
        inv["intermediate_steps"] = {
            "tool_calls": tool_calls or [],
            "tool_responses": tool_responses or [],
        }
    return inv


def _call(name, args=None):
    return {"name": name, "args": args or {}}


def _resp(name, output="ok", status=None):
    r = {"name": name, "output": output}
    if status:
        r["status"] = status
    return r


class TestToolEfficiencyBasic:
    def test_no_invocations(self):
        result = _run(_make_input([]))
        assert result["status"] == "NOT_EVALUATED"
        assert result["score"] == 0.0

    def test_no_tool_calls_optional(self):
        result = _run(_make_input([_inv("inv-1", tool_calls=[])]))
        assert result["score"] == 1.0

    def test_no_tool_calls_required(self):
        result = _run(_make_input(
            [_inv("inv-1", tool_calls=[])],
            config={"min_tool_calls": 1},
        ))
        assert result["score"] == 0.0

    def test_single_useful_call(self):
        result = _run(_make_input([
            _inv("inv-1",
                 tool_calls=[_call("search", {"q": "hello"})],
                 tool_responses=[_resp("search")]),
        ]))
        assert result["score"] == 1.0

    def test_no_intermediate_steps(self):
        inv = {"invocation_id": "inv-1"}
        result = _run(_make_input([inv]))
        assert result["score"] == 1.0  # no tool calls, tools optional


class TestToolEfficiencyDuplicates:
    def test_duplicate_calls_penalized(self):
        calls = [_call("search", {"q": "hello"})] * 3
        result = _run(_make_input([
            _inv("inv-1", tool_calls=calls),
        ]))
        # total=3, dupes=2, useful=1, efficiency=1/3, budget_factor=1.0
        assert result["score"] == pytest.approx(1 / 3, abs=0.01)

    def test_duplicates_not_penalized_when_disabled(self):
        calls = [_call("search", {"q": "hello"})] * 3
        result = _run(_make_input(
            [_inv("inv-1", tool_calls=calls)],
            config={"penalize_duplicates": False},
        ))
        # No dupe penalty: useful=3, efficiency=1.0, budget_factor=1.0
        assert result["score"] == 1.0

    def test_different_args_not_duplicates(self):
        calls = [
            _call("search", {"q": "hello"}),
            _call("search", {"q": "world"}),
        ]
        result = _run(_make_input([
            _inv("inv-1", tool_calls=calls),
        ]))
        assert result["score"] == 1.0


class TestToolEfficiencyErrors:
    def test_error_responses_penalized(self):
        calls = [_call("fetch"), _call("fetch")]
        responses = [
            _resp("fetch", status="error"),
            _resp("fetch", output="data"),
        ]
        result = _run(_make_input([
            _inv("inv-1", tool_calls=calls, tool_responses=responses),
        ]))
        # total=2, errors=1, useful=1, efficiency=0.5
        assert result["score"] < 1.0

    def test_errors_not_penalized_when_disabled(self):
        calls = [_call("fetch")]
        responses = [_resp("fetch", status="error")]
        result = _run(_make_input(
            [_inv("inv-1", tool_calls=calls, tool_responses=responses)],
            config={"penalize_errors": False},
        ))
        assert result["score"] == 1.0

    def test_failed_status(self):
        calls = [_call("run")]
        responses = [_resp("run", status="failed")]
        result = _run(_make_input([
            _inv("inv-1", tool_calls=calls, tool_responses=responses),
        ]))
        # useful=0, efficiency=0
        assert result["score"] == 0.0


class TestToolEfficiencyBudget:
    def test_over_budget(self):
        calls = [_call(f"tool-{i}") for i in range(20)]
        result = _run(_make_input(
            [_inv("inv-1", tool_calls=calls)],
            config={"max_tool_calls": 15},
        ))
        # Over budget -> budget_factor < 1.0
        assert result["score"] < 1.0

    def test_zero_max_tool_calls_guard(self):
        """Regression: division by zero when max_tool_calls=0."""
        calls = [_call("search")]
        result = _run(_make_input(
            [_inv("inv-1", tool_calls=calls)],
            config={"max_tool_calls": 0},
        ))
        assert result["score"] == 0.0


class TestToolEfficiencyMultipleInvocations:
    def test_average_across_invocations(self):
        result = _run(_make_input([
            _inv("inv-1", tool_calls=[_call("a")]),                          # score ≈ 1.0
            _inv("inv-2", tool_calls=[_call("b")],
                 tool_responses=[_resp("b", status="error")]),               # score = 0.0
        ]))
        assert result["score"] == pytest.approx(0.5, abs=0.01)
        assert len(result["per_invocation_scores"]) == 2

    def test_uses_issues_key_in_details(self):
        result = _run(_make_input([
            _inv("inv-1", tool_calls=[_call("a")]),
        ]))
        assert "issues" in result["details"]


class TestToolEfficiencyAttributeAccess:
    """Verify evaluator works with SDK model objects (attribute access, not dict)."""

    def test_tool_call_with_empty_args(self):
        result = _run(_make_input([
            _inv("inv-1", tool_calls=[_call("ping")]),
        ]))
        assert result["score"] == 1.0

    def test_tool_response_with_none_status(self):
        responses = [_resp("ping", output="pong")]
        result = _run(_make_input([
            _inv("inv-1", tool_calls=[_call("ping")], tool_responses=responses),
        ]))
        assert result["score"] == 1.0
