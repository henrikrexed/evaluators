"""End-to-end tests for the token_efficiency evaluator."""

import json
import subprocess
import sys

import pytest

EVALUATOR = "evaluators/token_efficiency/token_efficiency.py"


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
        "metric_name": "token_efficiency",
        "threshold": 0.5,
        "config": config or {},
        "invocations": invocations,
    }


def _inv(inv_id, perf=None):
    return {"invocation_id": inv_id, "performance_metrics": perf}


class TestTokenEfficiencyBasic:
    def test_no_invocations(self):
        result = _run(_make_input([]))
        assert result["status"] == "NOT_EVALUATED"
        assert result["score"] == 0.0

    def test_no_token_data(self):
        result = _run(_make_input([_inv("inv-1", {})]))
        assert result["status"] == "NOT_EVALUATED"
        assert result["score"] == 0.0

    def test_no_perf_metrics(self):
        result = _run(_make_input([_inv("inv-1", None)]))
        assert result["status"] == "NOT_EVALUATED"

    def test_perfect_score_zero_tokens(self):
        """Zero tokens should give score 1.0 — not be treated as missing data."""
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "output_tokens": 0}),
        ]))
        assert result["score"] == 1.0
        assert result["status"] is None

    def test_half_budget(self):
        result = _run(_make_input(
            [_inv("inv-1", {"input_tokens": 75000, "output_tokens": 25000})],
        ))
        assert result["score"] == pytest.approx(0.5)

    def test_over_budget_clamps_to_zero(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 200000, "output_tokens": 60000}),
        ]))
        assert result["score"] == 0.0


class TestTokenEfficiencyZeroValues:
    """Regression: `or` operator previously dropped zero token values."""

    def test_zero_input_tokens_not_dropped(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "output_tokens": 10000}),
        ]))
        assert result["score"] > 0.0

    def test_zero_output_tokens_not_dropped(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 10000, "output_tokens": 0}),
        ]))
        assert result["score"] > 0.0

    def test_zero_input_not_fallback_to_prompt(self):
        """When input_tokens=0, it should NOT fall back to prompt_tokens."""
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "prompt_tokens": 999999, "output_tokens": 100}),
        ]))
        # Score should be high because input_tokens=0, not 999999
        assert result["score"] > 0.9


class TestTokenEfficiencyAliases:
    def test_prompt_tokens_alias(self):
        result = _run(_make_input([
            _inv("inv-1", {"prompt_tokens": 75000, "completion_tokens": 25000}),
        ]))
        assert result["score"] == pytest.approx(0.5)

    def test_input_tokens_takes_precedence(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "prompt_tokens": 150000,
                           "output_tokens": 0, "completion_tokens": 50000}),
        ]))
        # input_tokens=0 should be used, not prompt_tokens=150000
        assert result["score"] == 1.0


class TestTokenEfficiencyConfig:
    def test_custom_budget(self):
        result = _run(_make_input(
            [_inv("inv-1", {"input_tokens": 500, "output_tokens": 250})],
            config={"max_input_tokens": 1000, "max_output_tokens": 500},
        ))
        assert result["score"] == pytest.approx(0.5)

    def test_zero_max_input_gives_perfect(self):
        result = _run(_make_input(
            [_inv("inv-1", {"input_tokens": 100, "output_tokens": 0})],
            config={"max_input_tokens": 0},
        ))
        # max_input=0 -> input_score=1.0, output_score=1.0
        assert result["score"] == 1.0


class TestTokenEfficiencyMultipleInvocations:
    def test_average_across_invocations(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "output_tokens": 0}),       # score = 1.0
            _inv("inv-2", {"input_tokens": 150000, "output_tokens": 50000}),  # score = 0.0
        ]))
        assert result["score"] == pytest.approx(0.5)
        assert len(result["per_invocation_scores"]) == 2

    def test_mixed_with_missing(self):
        result = _run(_make_input([
            _inv("inv-1", {"input_tokens": 0, "output_tokens": 0}),  # score = 1.0
            _inv("inv-2", None),  # score = 0.0 (missing)
        ]))
        # has_data is true from inv-1, so not NOT_EVALUATED
        assert result["status"] is None
        assert result["score"] == pytest.approx(0.5)
